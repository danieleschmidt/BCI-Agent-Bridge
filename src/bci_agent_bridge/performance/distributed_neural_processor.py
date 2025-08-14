"""
Distributed Neural Processing System for high-throughput BCI applications.
Implements load balancing, parallel processing, and intelligent resource management.
"""

import asyncio
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import multiprocessing as mp
import queue
import uuid
import json
from datetime import datetime

# Import core components
try:
    from ..core.bridge import NeuralData, DecodedIntention
    from ..decoders.base import BaseDecoder
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

# Import performance monitoring
try:
    from .load_balancer import LoadBalancer, WorkerNode
    from .auto_scaler import AutoScaler
    from .distributed_cache import DistributedCache
    _PERFORMANCE_AVAILABLE = True
except ImportError:
    _PERFORMANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Neural processing modes for different performance profiles."""
    SINGLE_THREADED = "single_threaded"
    MULTI_THREADED = "multi_threaded"
    MULTI_PROCESS = "multi_process"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


class ProcessorState(Enum):
    """Processor state management."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    SCALING = "scaling"
    DEGRADED = "degraded"
    SHUTDOWN = "shutdown"


@dataclass
class ProcessingTask:
    """Individual neural processing task."""
    id: str
    neural_data: NeuralData
    priority: int = 1  # 1=low, 5=critical
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[DecodedIntention] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkerStats:
    """Worker performance statistics."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_processing_time_ms: float = 0.0
    current_load: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    active_tasks: int = 0
    total_uptime: float = 0.0


class NeuralProcessingWorker:
    """Individual worker for neural data processing."""
    
    def __init__(self, worker_id: str, decoder_config: Dict[str, Any]):
        self.worker_id = worker_id
        self.decoder_config = decoder_config
        self.stats = WorkerStats(worker_id=worker_id)
        self.is_running = False
        self.task_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        
        # Initialize decoder (would be loaded based on config)
        self.decoder = None
        self._initialize_decoder()
        
        logger.info(f"Neural processing worker {worker_id} initialized")
    
    def _initialize_decoder(self) -> None:
        """Initialize the neural decoder for this worker."""
        try:
            # In real implementation, would load specific decoder based on config
            from ..decoders.p300 import P300Decoder
            self.decoder = P300Decoder(
                channels=self.decoder_config.get('channels', 8),
                sampling_rate=self.decoder_config.get('sampling_rate', 250)
            )
            logger.info(f"Decoder initialized for worker {self.worker_id}")
        except Exception as e:
            logger.error(f"Failed to initialize decoder for worker {self.worker_id}: {e}")
    
    def start(self) -> None:
        """Start the worker processing loop."""
        self.is_running = True
        self.stats.total_uptime = time.time()
        
        while self.is_running:
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1.0)
                self._process_task(task)
                self.task_queue.task_done()
                
            except queue.Empty:
                continue  # No tasks available
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                self.stats.tasks_failed += 1
    
    def _process_task(self, task: ProcessingTask) -> None:
        """Process individual neural processing task."""
        start_time = time.time()
        task.started_at = start_time
        task.worker_id = self.worker_id
        self.stats.active_tasks += 1
        
        try:
            if self.decoder is None:
                raise RuntimeError("Decoder not initialized")
            
            # Extract features and predict
            features = self.decoder.extract_features(task.neural_data.data)
            prediction = self.decoder.predict(features)
            confidence = self.decoder.get_confidence()
            
            # Create result
            task.result = DecodedIntention(
                command=self._map_prediction_to_command(prediction),
                confidence=confidence,
                context={
                    "paradigm": self.decoder_config.get('paradigm', 'P300'),
                    "prediction": prediction,
                    "worker_id": self.worker_id,
                    "processing_time_ms": (time.time() - start_time) * 1000
                },
                timestamp=time.time()
            )
            
            task.completed_at = time.time()
            self.stats.tasks_completed += 1
            
            # Update performance stats
            processing_time = (task.completed_at - start_time) * 1000
            self.stats.avg_processing_time_ms = (
                (self.stats.avg_processing_time_ms * (self.stats.tasks_completed - 1) + processing_time) /
                self.stats.tasks_completed
            )
            
        except Exception as e:
            task.error = str(e)
            task.completed_at = time.time()
            self.stats.tasks_failed += 1
            logger.error(f"Task processing failed in worker {self.worker_id}: {e}")
        
        finally:
            self.stats.active_tasks -= 1
            self.stats.last_heartbeat = time.time()
            
            # Put result in result queue
            self.result_queue.put(task)
    
    def _map_prediction_to_command(self, prediction: Any) -> str:
        """Map decoder prediction to command."""
        paradigm = self.decoder_config.get('paradigm', 'P300')
        
        if paradigm == 'P300':
            return "Select current item" if prediction == 1 else "No selection"
        elif paradigm == 'MotorImagery':
            movement_map = {0: "Move left", 1: "Move right", 2: "Move forward", 3: "Move backward"}
            return movement_map.get(prediction, "Unknown movement")
        elif paradigm == 'SSVEP':
            freq_map = {0: "Option 1", 1: "Option 2", 2: "Option 3", 3: "Option 4"}
            return freq_map.get(prediction, "No selection")
        
        return "Unknown command"
    
    def stop(self) -> None:
        """Stop the worker."""
        self.is_running = False
        self.stats.total_uptime = time.time() - self.stats.total_uptime
    
    def add_task(self, task: ProcessingTask) -> bool:
        """Add task to worker queue."""
        try:
            self.task_queue.put_nowait(task)
            return True
        except queue.Full:
            return False
    
    def get_result(self) -> Optional[ProcessingTask]:
        """Get completed task result."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


class DistributedNeuralProcessor:
    """
    High-performance distributed neural processing system with intelligent
    load balancing, auto-scaling, and resource optimization.
    """
    
    def __init__(
        self,
        processing_mode: ProcessingMode = ProcessingMode.MULTI_THREADED,
        max_workers: int = None,
        enable_auto_scaling: bool = True,
        enable_caching: bool = True,
        quality_threshold: float = 0.5
    ):
        self.processing_mode = processing_mode
        self.max_workers = max_workers or self._calculate_optimal_workers()
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_caching = enable_caching
        self.quality_threshold = quality_threshold
        
        # State management
        self.state = ProcessorState.INITIALIZING
        self.workers = {}  # worker_id -> NeuralProcessingWorker
        self.worker_threads = {}  # worker_id -> Thread
        self.worker_processes = {}  # worker_id -> Process
        
        # Task management
        self.pending_tasks = queue.PriorityQueue()
        self.active_tasks = {}  # task_id -> ProcessingTask
        self.completed_tasks = deque(maxlen=10000)
        self.failed_tasks = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_processing_time_ms': 0.0,
            'throughput_tasks_per_sec': 0.0,
            'current_load': 0.0,
            'peak_load': 0.0
        }
        
        # Load balancer and auto-scaler
        if _PERFORMANCE_AVAILABLE:
            self.load_balancer = LoadBalancer()
            if self.enable_auto_scaling:
                self.auto_scaler = AutoScaler(
                    min_workers=1,
                    max_workers=self.max_workers,
                    target_cpu_percent=70.0
                )
        
        # Distributed cache for processed results
        if self.enable_caching and _PERFORMANCE_AVAILABLE:
            self.cache = DistributedCache(max_size=10000, ttl_seconds=3600)
        else:
            self.cache = None
        
        # Monitoring and management
        self.executor = None
        self.is_running = False
        self.management_thread = None
        
        self._initialize_system()
        
        logger.info(f"Distributed Neural Processor initialized with {self.processing_mode.value} mode")
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources."""
        cpu_count = mp.cpu_count()
        
        if self.processing_mode == ProcessingMode.SINGLE_THREADED:
            return 1
        elif self.processing_mode == ProcessingMode.MULTI_THREADED:
            return min(cpu_count * 2, 16)  # 2x CPU cores, max 16
        elif self.processing_mode in [ProcessingMode.MULTI_PROCESS, ProcessingMode.DISTRIBUTED]:
            return min(cpu_count, 8)  # 1x CPU cores, max 8
        else:  # HYBRID
            return min(cpu_count + 2, 12)
    
    def _initialize_system(self) -> None:
        """Initialize the distributed processing system."""
        try:
            # Initialize executor based on processing mode
            if self.processing_mode == ProcessingMode.SINGLE_THREADED:
                self.executor = None  # Process directly
            elif self.processing_mode == ProcessingMode.MULTI_THREADED:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            elif self.processing_mode == ProcessingMode.MULTI_PROCESS:
                self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            
            # Initialize workers for custom management
            if self.processing_mode in [ProcessingMode.DISTRIBUTED, ProcessingMode.HYBRID]:
                self._initialize_workers()
            
            self.state = ProcessorState.IDLE
            logger.info("Distributed processing system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed processing system: {e}")
            self.state = ProcessorState.DEGRADED
    
    def _initialize_workers(self) -> None:
        """Initialize custom worker processes/threads."""
        decoder_config = {
            'channels': 8,
            'sampling_rate': 250,
            'paradigm': 'P300'
        }
        
        for i in range(self.max_workers):
            worker_id = f"worker_{i}"
            worker = NeuralProcessingWorker(worker_id, decoder_config)
            self.workers[worker_id] = worker
            
            if self.processing_mode == ProcessingMode.DISTRIBUTED:
                # Use process for true distributed processing
                process = mp.Process(target=worker.start)
                self.worker_processes[worker_id] = process
            else:
                # Use thread for hybrid mode
                thread = threading.Thread(target=worker.start, daemon=True)
                self.worker_threads[worker_id] = thread
    
    async def start(self) -> None:
        """Start the distributed processing system."""
        if self.is_running:
            logger.warning("Distributed processor already running")
            return
        
        self.is_running = True
        
        # Start workers
        for worker_id, worker in self.workers.items():
            if worker_id in self.worker_processes:
                self.worker_processes[worker_id].start()
            elif worker_id in self.worker_threads:
                self.worker_threads[worker_id].start()
        
        # Start management thread
        self.management_thread = threading.Thread(target=self._management_loop, daemon=True)
        self.management_thread.start()
        
        # Start auto-scaler if enabled
        if self.enable_auto_scaling and hasattr(self, 'auto_scaler'):
            await self.auto_scaler.start()
        
        self.state = ProcessorState.IDLE
        logger.info("Distributed neural processor started")
    
    def stop(self) -> None:
        """Stop the distributed processing system."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.state = ProcessorState.SHUTDOWN
        
        # Stop workers
        for worker in self.workers.values():
            worker.stop()
        
        # Stop processes/threads
        for process in self.worker_processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        
        for thread in self.worker_threads.values():
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Stop executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Stop auto-scaler
        if self.enable_auto_scaling and hasattr(self, 'auto_scaler'):
            self.auto_scaler.stop()
        
        logger.info("Distributed neural processor stopped")
    
    def _management_loop(self) -> None:
        """Management loop for monitoring and optimization."""
        while self.is_running:
            try:
                # Collect results from workers
                self._collect_worker_results()
                
                # Update performance statistics
                self._update_performance_stats()
                
                # Check for auto-scaling triggers
                if self.enable_auto_scaling:
                    self._check_scaling_triggers()
                
                # Cleanup old tasks
                self._cleanup_old_tasks()
                
                # Health monitoring
                self._monitor_system_health()
                
                time.sleep(1.0)  # Management cycle interval
                
            except Exception as e:
                logger.error(f"Management loop error: {e}")
    
    def _collect_worker_results(self) -> None:
        """Collect completed task results from workers."""
        for worker in self.workers.values():
            while True:
                result = worker.get_result()
                if result is None:
                    break
                
                # Process the result
                if result.error:
                    self.failed_tasks.append(result)
                    self.performance_stats['failed_tasks'] += 1
                    
                    # Retry if possible
                    if result.retry_count < result.max_retries:
                        result.retry_count += 1
                        result.error = None
                        result.started_at = None
                        result.completed_at = None
                        self._submit_task_internal(result)
                else:
                    self.completed_tasks.append(result)
                    self.performance_stats['completed_tasks'] += 1
                
                # Remove from active tasks
                if result.id in self.active_tasks:
                    del self.active_tasks[result.id]
    
    def _update_performance_stats(self) -> None:
        """Update system performance statistics."""
        current_time = time.time()
        
        # Calculate throughput
        completed_in_last_minute = sum(
            1 for task in self.completed_tasks
            if current_time - task.completed_at < 60.0
        )
        self.performance_stats['throughput_tasks_per_sec'] = completed_in_last_minute / 60.0
        
        # Calculate average processing time
        if self.completed_tasks:
            recent_tasks = [
                task for task in list(self.completed_tasks)[-100:]
                if task.started_at and task.completed_at
            ]
            if recent_tasks:
                avg_time = np.mean([
                    (task.completed_at - task.started_at) * 1000
                    for task in recent_tasks
                ])
                self.performance_stats['avg_processing_time_ms'] = avg_time
        
        # Calculate current load
        active_task_count = len(self.active_tasks)
        max_concurrent = self.max_workers * 2  # Assume 2 tasks per worker max
        self.performance_stats['current_load'] = min(active_task_count / max_concurrent, 1.0)
        
        # Update peak load
        if self.performance_stats['current_load'] > self.performance_stats['peak_load']:
            self.performance_stats['peak_load'] = self.performance_stats['current_load']
    
    def _check_scaling_triggers(self) -> None:
        """Check if auto-scaling should be triggered."""
        if not hasattr(self, 'auto_scaler'):
            return
        
        current_load = self.performance_stats['current_load']
        
        # Scale up if high load
        if current_load > 0.8 and len(self.workers) < self.max_workers:
            self._scale_up()
        
        # Scale down if low load
        elif current_load < 0.3 and len(self.workers) > 1:
            self._scale_down()
    
    def _scale_up(self) -> None:
        """Add additional workers."""
        if len(self.workers) >= self.max_workers:
            return
        
        self.state = ProcessorState.SCALING
        
        try:
            worker_id = f"worker_{len(self.workers)}"
            decoder_config = {'channels': 8, 'sampling_rate': 250, 'paradigm': 'P300'}
            
            worker = NeuralProcessingWorker(worker_id, decoder_config)
            self.workers[worker_id] = worker
            
            # Start the worker
            if self.processing_mode == ProcessingMode.DISTRIBUTED:
                process = mp.Process(target=worker.start)
                self.worker_processes[worker_id] = process
                process.start()
            else:
                thread = threading.Thread(target=worker.start, daemon=True)
                self.worker_threads[worker_id] = thread
                thread.start()
            
            logger.info(f"Scaled up: added worker {worker_id}")
            
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
        finally:
            self.state = ProcessorState.PROCESSING if self.active_tasks else ProcessorState.IDLE
    
    def _scale_down(self) -> None:
        """Remove workers when load is low."""
        if len(self.workers) <= 1:
            return
        
        self.state = ProcessorState.SCALING
        
        try:
            # Find worker with least load
            worker_id = min(
                self.workers.keys(),
                key=lambda wid: self.workers[wid].stats.active_tasks
            )
            
            # Stop and remove worker
            worker = self.workers[worker_id]
            worker.stop()
            
            if worker_id in self.worker_processes:
                process = self.worker_processes[worker_id]
                process.terminate()
                process.join(timeout=5.0)
                del self.worker_processes[worker_id]
            
            if worker_id in self.worker_threads:
                thread = self.worker_threads[worker_id]
                thread.join(timeout=5.0)
                del self.worker_threads[worker_id]
            
            del self.workers[worker_id]
            logger.info(f"Scaled down: removed worker {worker_id}")
            
        except Exception as e:
            logger.error(f"Failed to scale down: {e}")
        finally:
            self.state = ProcessorState.PROCESSING if self.active_tasks else ProcessorState.IDLE
    
    def _cleanup_old_tasks(self) -> None:
        """Clean up old completed and failed tasks."""
        current_time = time.time()
        cleanup_age = 3600.0  # 1 hour
        
        # Clean up old completed tasks (keep in deque with maxlen)
        # Deque automatically handles size limit
        
        # Clean up old active tasks that might be stuck
        stuck_tasks = [
            task_id for task_id, task in self.active_tasks.items()
            if current_time - task.submitted_at > 300.0  # 5 minutes
        ]
        
        for task_id in stuck_tasks:
            task = self.active_tasks[task_id]
            task.error = "Task timeout"
            self.failed_tasks.append(task)
            del self.active_tasks[task_id]
            logger.warning(f"Cleaned up stuck task: {task_id}")
    
    def _monitor_system_health(self) -> None:
        """Monitor overall system health."""
        try:
            # Check worker health
            healthy_workers = 0
            for worker in self.workers.values():
                if time.time() - worker.stats.last_heartbeat < 60.0:  # Heartbeat within 1 minute
                    healthy_workers += 1
            
            # Update state based on health
            if healthy_workers == 0:
                self.state = ProcessorState.DEGRADED
            elif healthy_workers < len(self.workers) * 0.5:  # Less than 50% healthy
                if self.state == ProcessorState.IDLE:
                    self.state = ProcessorState.DEGRADED
            elif self.state == ProcessorState.DEGRADED and healthy_workers >= len(self.workers) * 0.8:
                self.state = ProcessorState.IDLE
            
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
    
    async def process_neural_data(
        self,
        neural_data: NeuralData,
        priority: int = 1,
        timeout_seconds: float = 30.0
    ) -> Optional[DecodedIntention]:
        """
        Process neural data with distributed processing.
        
        Args:
            neural_data: Neural signal data to process
            priority: Task priority (1=low, 5=critical)
            timeout_seconds: Maximum time to wait for result
            
        Returns:
            Decoded intention or None if timeout/error
        """
        if not self.is_running:
            raise RuntimeError("Distributed processor not running")
        
        # Check cache first if enabled
        if self.cache:
            cache_key = self._generate_cache_key(neural_data)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for neural data processing")
                return cached_result
        
        # Create processing task
        task = ProcessingTask(
            id=str(uuid.uuid4()),
            neural_data=neural_data,
            priority=priority
        )
        
        # Submit task for processing
        result = await self._submit_and_wait(task, timeout_seconds)
        
        # Cache result if successful and caching enabled
        if result and result.result and self.cache:
            cache_key = self._generate_cache_key(neural_data)
            await self.cache.set(cache_key, result.result, ttl=3600)
        
        return result.result if result else None
    
    async def _submit_and_wait(self, task: ProcessingTask, timeout_seconds: float) -> Optional[ProcessingTask]:
        """Submit task and wait for completion."""
        self.state = ProcessorState.PROCESSING
        
        # Submit task
        success = self._submit_task_internal(task)
        if not success:
            logger.error(f"Failed to submit task {task.id}")
            return None
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            # Check if task completed
            if task.id not in self.active_tasks:
                # Task completed, find it in results
                for completed_task in list(self.completed_tasks):
                    if completed_task.id == task.id:
                        return completed_task
                
                for failed_task in list(self.failed_tasks):
                    if failed_task.id == task.id:
                        return failed_task
            
            await asyncio.sleep(0.01)  # Small delay
        
        # Timeout occurred
        logger.warning(f"Task {task.id} timed out after {timeout_seconds}s")
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        return None
    
    def _submit_task_internal(self, task: ProcessingTask) -> bool:
        """Submit task to available worker."""
        self.active_tasks[task.id] = task
        self.performance_stats['total_tasks'] += 1
        
        # Find best worker (least loaded)
        if not self.workers:
            logger.error("No workers available")
            return False
        
        best_worker = min(
            self.workers.values(),
            key=lambda w: w.stats.active_tasks
        )
        
        success = best_worker.add_task(task)
        if not success:
            logger.warning(f"Worker {best_worker.worker_id} queue full")
            # Try other workers
            for worker in self.workers.values():
                if worker != best_worker:
                    success = worker.add_task(task)
                    if success:
                        break
        
        return success
    
    def _generate_cache_key(self, neural_data: NeuralData) -> str:
        """Generate cache key for neural data."""
        # Create hash based on data content and parameters
        data_hash = hash(neural_data.data.tobytes())
        param_hash = hash((
            neural_data.sampling_rate,
            len(neural_data.channels),
            neural_data.metadata.get('paradigm', 'unknown')
        ))
        
        return f"neural_processing_{data_hash}_{param_hash}"
    
    async def process_batch(
        self,
        neural_data_batch: List[NeuralData],
        batch_priority: int = 1
    ) -> List[Optional[DecodedIntention]]:
        """
        Process a batch of neural data in parallel.
        
        Args:
            neural_data_batch: List of neural data to process
            batch_priority: Priority for all tasks in batch
            
        Returns:
            List of decoded intentions (same order as input)
        """
        if not neural_data_batch:
            return []
        
        logger.info(f"Processing batch of {len(neural_data_batch)} neural data samples")
        
        # Submit all tasks
        tasks = []
        for i, neural_data in enumerate(neural_data_batch):
            task = ProcessingTask(
                id=f"batch_{uuid.uuid4()}_{i}",
                neural_data=neural_data,
                priority=batch_priority
            )
            tasks.append(task)
            self._submit_task_internal(task)
        
        # Wait for all tasks to complete
        results = [None] * len(tasks)
        completed_count = 0
        start_time = time.time()
        timeout = 60.0  # 1 minute timeout for batch
        
        while completed_count < len(tasks) and time.time() - start_time < timeout:
            for i, task in enumerate(tasks):
                if results[i] is None and task.id not in self.active_tasks:
                    # Find completed task
                    for completed_task in list(self.completed_tasks):
                        if completed_task.id == task.id:
                            results[i] = completed_task.result
                            completed_count += 1
                            break
                    
                    if results[i] is None:
                        # Check failed tasks
                        for failed_task in list(self.failed_tasks):
                            if failed_task.id == task.id:
                                results[i] = None  # Failed task
                                completed_count += 1
                                break
            
            await asyncio.sleep(0.01)
        
        logger.info(f"Batch processing completed: {completed_count}/{len(tasks)} tasks")
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        worker_stats = {}
        for worker_id, worker in self.workers.items():
            worker_stats[worker_id] = {
                'tasks_completed': worker.stats.tasks_completed,
                'tasks_failed': worker.stats.tasks_failed,
                'avg_processing_time_ms': worker.stats.avg_processing_time_ms,
                'active_tasks': worker.stats.active_tasks,
                'cpu_usage_percent': worker.stats.cpu_usage_percent,
                'memory_usage_mb': worker.stats.memory_usage_mb,
                'last_heartbeat': worker.stats.last_heartbeat
            }
        
        return {
            'system_stats': self.performance_stats,
            'worker_stats': worker_stats,
            'system_state': self.state.value,
            'processing_mode': self.processing_mode.value,
            'active_workers': len(self.workers),
            'max_workers': self.max_workers,
            'queue_sizes': {
                'pending': self.pending_tasks.qsize(),
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks)
            },
            'cache_stats': self.cache.get_stats() if self.cache else None,
            'auto_scaling_enabled': self.enable_auto_scaling
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'state': self.state.value,
            'is_running': self.is_running,
            'processing_mode': self.processing_mode.value,
            'workers': {
                'active': len(self.workers),
                'healthy': sum(
                    1 for w in self.workers.values()
                    if time.time() - w.stats.last_heartbeat < 60.0
                ),
                'max': self.max_workers
            },
            'performance': {
                'throughput_tasks_per_sec': self.performance_stats['throughput_tasks_per_sec'],
                'avg_processing_time_ms': self.performance_stats['avg_processing_time_ms'],
                'current_load': self.performance_stats['current_load'],
                'success_rate': (
                    self.performance_stats['completed_tasks'] /
                    max(self.performance_stats['total_tasks'], 1)
                ) * 100
            },
            'features': {
                'auto_scaling': self.enable_auto_scaling,
                'caching': self.enable_caching,
                'load_balancing': _PERFORMANCE_AVAILABLE
            }
        }


# Factory function for easy instantiation
def create_distributed_processor(config: Optional[Dict[str, Any]] = None) -> DistributedNeuralProcessor:
    """Create and configure a distributed neural processor."""
    config = config or {}
    
    mode_map = {
        'single': ProcessingMode.SINGLE_THREADED,
        'thread': ProcessingMode.MULTI_THREADED,
        'process': ProcessingMode.MULTI_PROCESS,
        'distributed': ProcessingMode.DISTRIBUTED,
        'hybrid': ProcessingMode.HYBRID
    }
    
    processing_mode = mode_map.get(
        config.get('mode', 'thread'),
        ProcessingMode.MULTI_THREADED
    )
    
    return DistributedNeuralProcessor(
        processing_mode=processing_mode,
        max_workers=config.get('max_workers'),
        enable_auto_scaling=config.get('auto_scaling', True),
        enable_caching=config.get('caching', True),
        quality_threshold=config.get('quality_threshold', 0.5)
    )