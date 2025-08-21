"""
Optimized BCI Bridge with performance enhancements for Generation 3.
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, AsyncGenerator, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from .bridge import BCIBridge, NeuralData, DecodedIntention
from ..performance.caching import NeuralDataCache, CachePolicy
from ..performance.batch_processor import BatchProcessor, BatchItem, BatchConfig, BatchStrategy
from ..performance.connection_pool import ConnectionPool
from ..monitoring.metrics_collector import BCIMetricsCollector


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    enable_caching: bool = True
    enable_batching: bool = True
    enable_parallel_processing: bool = True
    cache_size_mb: int = 50
    batch_size: int = 32
    batch_timeout_ms: int = 100
    worker_threads: int = 4
    prefetch_samples: int = 1000


class OptimizedBCIBridge(BCIBridge):
    """
    Performance-optimized BCI Bridge with caching, batching, and parallel processing.
    """
    
    def __init__(self, 
                 device: str = "Simulation",
                 channels: int = 8,
                 sampling_rate: int = 250,
                 paradigm: str = "P300",
                 buffer_size: int = 1000,
                 privacy_mode: bool = True,
                 performance_config: Optional[PerformanceConfig] = None):
        
        super().__init__(device, channels, sampling_rate, paradigm, buffer_size, privacy_mode)
        
        self.perf_config = performance_config or PerformanceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance components
        self._setup_performance_components()
        
        # Metrics for monitoring
        self.metrics = None  # Can be initialized later if needed
        
        # Threading and execution
        self.executor = ThreadPoolExecutor(max_workers=self.perf_config.worker_threads)
        self._processing_lock = threading.Lock()
        
        # Performance tracking
        self._total_processed = 0
        self._processing_times = []
        self._cache_stats = {"hits": 0, "misses": 0}
        
        self.logger.info(f"OptimizedBCIBridge initialized with performance config: {self.perf_config}")
    
    def _setup_performance_components(self):
        """Initialize performance optimization components."""
        
        # Intelligent caching for neural features and decoder results
        if self.perf_config.enable_caching:
            self.cache = NeuralDataCache(
                max_size_bytes=self.perf_config.cache_size_mb * 1024 * 1024
            )
        else:
            self.cache = None
        
        # Batch processor for efficient neural data processing
        if self.perf_config.enable_batching:
            batch_config = BatchConfig(
                max_batch_size=self.perf_config.batch_size,
                max_wait_time=self.perf_config.batch_timeout_ms / 1000.0,
                strategy=BatchStrategy.ADAPTIVE,
                parallel_workers=self.perf_config.worker_threads
            )
            
            self.batch_processor = BatchProcessor(
                processor_func=self._process_neural_batch,
                config=batch_config
            )
            self.batch_processor.start()
        else:
            self.batch_processor = None
    
    async def optimized_stream(self) -> AsyncGenerator[NeuralData, None]:
        """
        Optimized streaming with prefetching and parallel processing.
        """
        if not self._device_connected:
            raise RuntimeError("BCI device not connected")
        
        self.is_streaming = True
        self.logger.info("Starting optimized neural data stream")
        
        # Prefetch buffer for smooth streaming
        prefetch_queue = asyncio.Queue(maxsize=self.perf_config.prefetch_samples)
        
        # Producer task for data fetching
        async def data_producer():
            while self.is_streaming:
                try:
                    raw_data = await self._read_raw_data()
                    processed_data = self.preprocessor.process(raw_data)
                    
                    neural_data = NeuralData(
                        data=processed_data,
                        timestamp=time.time(),
                        channels=[f"CH{i+1}" for i in range(self.channels)],
                        sampling_rate=self.sampling_rate,
                        metadata={"device": self.device.value, "paradigm": self.paradigm.value}
                    )
                    
                    await prefetch_queue.put(neural_data)
                    
                except Exception as e:
                    self.logger.error(f"Error in data producer: {e}")
                    break
        
        # Start producer task
        producer_task = asyncio.create_task(data_producer())
        
        try:
            while self.is_streaming:
                # Get data from prefetch queue
                neural_data = await prefetch_queue.get()
                
                # Add to buffer with performance tracking
                processing_start = time.perf_counter()
                self._add_to_buffer_safe(neural_data)
                processing_time = time.perf_counter() - processing_start
                
                self._processing_times.append(processing_time)
                if len(self._processing_times) > 1000:
                    self._processing_times.pop(0)
                
                self._total_processed += 1
                
                yield neural_data
                
        except asyncio.CancelledError:
            self.logger.info("Optimized neural data stream cancelled")
        finally:
            self.is_streaming = False
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass
    
    def optimized_decode_intention(self, neural_data: NeuralData) -> DecodedIntention:
        """
        Optimized intention decoding with caching and parallel processing.
        """
        if self.decoder is None:
            raise RuntimeError("No decoder initialized")
        
        # Generate cache key for neural data
        cache_key = None
        cached_features = None
        if self.cache:
            data_hash = self._compute_data_hash(neural_data.data)
            cache_key = f"features_{self.paradigm.value}_{data_hash}"
            
            # Try to get features from cache
            cached_features = self.cache.get(cache_key)
            if cached_features is not None:
                features = cached_features
                self._cache_stats["hits"] += 1
            else:
                features = self.decoder.extract_features(neural_data.data)
                self.cache.put(cache_key, features, ttl=300)  # Cache for 5 minutes
                self._cache_stats["misses"] += 1
        else:
            features = self.decoder.extract_features(neural_data.data)
        
        # Batch processing for predictions
        if self.batch_processor and self.perf_config.enable_batching:
            # Add to batch processor for efficient prediction
            batch_item = BatchItem(
                data=features,
                timestamp=time.time(),
                metadata={"neural_data_id": id(neural_data)}
            )
            
            # For now, process immediately (could be async in real implementation)
            prediction = self.decoder.predict(features)
            confidence = self.decoder.get_confidence()
        else:
            prediction = self.decoder.predict(features)
            confidence = self.decoder.get_confidence()
        
        # Map prediction to command
        command = self._map_prediction_to_command(prediction)
        
        return DecodedIntention(
            command=command,
            confidence=confidence,
            context={
                "paradigm": self.paradigm.value,
                "prediction": prediction,
                "timestamp": neural_data.timestamp,
                "cache_hit": cached_features is not None if self.cache else False
            },
            timestamp=time.time(),
            neural_features=features if not self.privacy_mode else None
        )
    
    def _process_neural_batch(self, batch_items: List[BatchItem]) -> List[Any]:
        """Process a batch of neural features efficiently."""
        if not batch_items:
            return []
        
        try:
            # Extract features from batch items
            features_batch = [item.data for item in batch_items]
            
            # Process batch in parallel if enabled
            if self.perf_config.enable_parallel_processing:
                results = []
                with ThreadPoolExecutor(max_workers=self.perf_config.worker_threads) as executor:
                    futures = [executor.submit(self.decoder.predict, features) 
                              for features in features_batch]
                    
                    for future in futures:
                        results.append(future.result())
                
                return results
            else:
                # Sequential processing
                return [self.decoder.predict(features) for features in features_batch]
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            return [0] * len(batch_items)  # Return default predictions
    
    def _compute_data_hash(self, data: np.ndarray) -> str:
        """Compute hash for neural data for caching."""
        # Use a subset of data for hashing to balance speed vs uniqueness
        if data.size > 1000:
            # Sample every 10th point for large arrays
            sample_data = data.flatten()[::10]
        else:
            sample_data = data.flatten()
        
        # Convert to bytes and hash
        data_bytes = sample_data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:16]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        avg_processing_time = np.mean(self._processing_times) if self._processing_times else 0
        
        cache_hit_rate = 0
        if self._cache_stats["hits"] + self._cache_stats["misses"] > 0:
            cache_hit_rate = self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"])
        
        metrics = {
            "total_processed": self._total_processed,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "cache_hit_rate": cache_hit_rate,
            "cache_stats": self._cache_stats.copy(),
            "buffer_utilization": len(self.data_buffer) / self.buffer_size,
            "throughput_samples_per_sec": self._calculate_throughput(),
            "memory_usage_mb": self._estimate_memory_usage(),
            "performance_config": {
                "caching_enabled": self.perf_config.enable_caching,
                "batching_enabled": self.perf_config.enable_batching,
                "parallel_processing": self.perf_config.enable_parallel_processing,
                "worker_threads": self.perf_config.worker_threads
            }
        }
        
        return metrics
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput in samples per second."""
        if len(self._processing_times) < 2:
            return 0.0
        
        # Estimate based on recent processing times
        recent_times = self._processing_times[-100:]  # Last 100 samples
        avg_time = np.mean(recent_times)
        
        if avg_time > 0:
            return 1.0 / avg_time  # samples per second
        return 0.0
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        memory_mb = 0.0
        
        # Buffer memory
        if self.data_buffer:
            sample_size = self.data_buffer[0].data.nbytes if self.data_buffer else 0
            memory_mb += (sample_size * len(self.data_buffer)) / (1024 * 1024)
        
        # Cache memory
        if self.cache:
            stats = self.cache.get_stats()
            memory_mb += stats.get('size_bytes', 0) / (1024 * 1024)
        
        return memory_mb
    
    def optimize_for_latency(self):
        """Optimize configuration for low latency."""
        self.perf_config.batch_size = 1
        self.perf_config.batch_timeout_ms = 1
        self.perf_config.enable_batching = False
        self.logger.info("Optimized for low latency")
    
    def optimize_for_throughput(self):
        """Optimize configuration for high throughput."""
        self.perf_config.batch_size = 64
        self.perf_config.batch_timeout_ms = 50
        self.perf_config.enable_batching = True
        self.perf_config.enable_parallel_processing = True
        self.logger.info("Optimized for high throughput")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.batch_processor:
                self.batch_processor.stop()
        except Exception as e:
            self.logger.warning(f"Error stopping batch processor: {e}")
        
        if self.cache:
            self.cache.clear()
        
        self.executor.shutdown(wait=True)
        self.logger.info("OptimizedBCIBridge cleaned up")


# Import for easier access
import hashlib