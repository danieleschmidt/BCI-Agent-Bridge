"""
Batch processing system for efficient neural data handling.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


class BatchStrategy(Enum):
    TIME_BASED = "time_based"      # Process after time interval
    SIZE_BASED = "size_based"      # Process after reaching size
    HYBRID = "hybrid"              # Combination of time and size
    ADAPTIVE = "adaptive"          # Adaptive based on load


@dataclass
class BatchItem:
    data: Any
    timestamp: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None


@dataclass
class BatchConfig:
    max_batch_size: int = 100
    max_wait_time: float = 1.0
    strategy: BatchStrategy = BatchStrategy.HYBRID
    priority_levels: int = 3
    enable_compression: bool = False
    parallel_workers: int = 4


class BatchProcessor:
    """
    Generic batch processor for efficient data processing.
    """
    
    def __init__(self, 
                 processor_func: Callable[[List[BatchItem]], Any],
                 config: BatchConfig = None):
        self.processor_func = processor_func
        self.config = config or BatchConfig()
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.is_running = False
        self.batch_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.current_batch: List[BatchItem] = []
        self.last_flush_time = time.time()
        
        # Threading
        self._lock = threading.Lock()
        self._processor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Metrics
        self.total_items_processed = 0
        self.total_batches_processed = 0
        self.avg_batch_size = 0.0
        self.avg_processing_time = 0.0
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
    
    def start(self) -> None:
        """Start the batch processor."""
        if self.is_running:
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        self._processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processor_thread.start()
        self.logger.info("Batch processor started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop the batch processor."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._shutdown_event.set()
        
        # Wait for processor thread to finish
        if self._processor_thread:
            self._processor_thread.join(timeout)
            if self._processor_thread.is_alive():
                self.logger.warning("Processor thread did not stop gracefully")
        
        # Process remaining items
        self._flush_batch()
        
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=timeout)
        
        self.logger.info("Batch processor stopped")
    
    def add_item(self, data: Any, priority: int = 0, 
                 metadata: Dict[str, Any] = None, 
                 callback: Optional[Callable] = None) -> None:
        """Add item to batch queue."""
        if not self.is_running:
            raise RuntimeError("Batch processor is not running")
        
        item = BatchItem(
            data=data,
            timestamp=time.time(),
            priority=priority,
            metadata=metadata or {},
            callback=callback
        )
        
        # Use negative priority for max heap behavior
        self.batch_queue.put((-priority, time.time(), item))
    
    def _processing_loop(self) -> None:
        """Main processing loop."""
        while self.is_running and not self._shutdown_event.is_set():
            try:
                self._process_queue()
                time.sleep(0.01)  # Small sleep to prevent busy waiting
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
    
    def _process_queue(self) -> None:
        """Process items from queue into batches."""
        current_time = time.time()
        
        # Check if we should flush based on time
        should_flush_time = (current_time - self.last_flush_time) >= self.config.max_wait_time
        
        # Collect items from queue
        items_added = 0
        while not self.batch_queue.empty() and len(self.current_batch) < self.config.max_batch_size:
            try:
                _, _, item = self.batch_queue.get_nowait()
                with self._lock:
                    self.current_batch.append(item)
                items_added += 1
            except queue.Empty:
                break
        
        # Check if we should flush based on size
        should_flush_size = len(self.current_batch) >= self.config.max_batch_size
        
        # Determine if we should flush
        should_flush = False
        if self.config.strategy == BatchStrategy.TIME_BASED:
            should_flush = should_flush_time and len(self.current_batch) > 0
        elif self.config.strategy == BatchStrategy.SIZE_BASED:
            should_flush = should_flush_size
        elif self.config.strategy == BatchStrategy.HYBRID:
            should_flush = should_flush_time or should_flush_size
        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            should_flush = self._adaptive_flush_decision()
        
        if should_flush:
            self._flush_batch()
    
    def _adaptive_flush_decision(self) -> bool:
        """Make adaptive decision about when to flush."""
        if not self.current_batch:
            return False
        
        current_time = time.time()
        batch_age = current_time - self.current_batch[0].timestamp
        queue_size = self.batch_queue.qsize()
        
        # Flush if batch is getting old
        if batch_age > self.config.max_wait_time:
            return True
        
        # Flush if queue is building up (backpressure)
        if queue_size > self.config.max_batch_size * 2:
            return True
        
        # Flush if batch is reasonably sized and some time has passed
        if len(self.current_batch) >= self.config.max_batch_size // 2 and batch_age > self.config.max_wait_time / 2:
            return True
        
        return False
    
    def _flush_batch(self) -> None:
        """Flush current batch for processing."""
        with self._lock:
            if not self.current_batch:
                return
            
            batch_to_process = self.current_batch.copy()
            self.current_batch.clear()
            self.last_flush_time = time.time()
        
        # Submit batch for processing
        if batch_to_process:
            future = self.executor.submit(self._process_batch, batch_to_process)
            # Don't wait for completion to avoid blocking
    
    def _process_batch(self, batch: List[BatchItem]) -> None:
        """Process a batch of items."""
        start_time = time.time()
        
        try:
            # Call the processor function
            result = self.processor_func(batch)
            
            # Execute callbacks
            for item in batch:
                if item.callback:
                    try:
                        item.callback(item, result)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(len(batch), processing_time)
            
            self.logger.debug(f"Processed batch of {len(batch)} items in {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            
            # Execute error callbacks
            for item in batch:
                if item.callback:
                    try:
                        item.callback(item, None, e)
                    except Exception as callback_error:
                        self.logger.error(f"Error callback error: {callback_error}")
    
    def _update_metrics(self, batch_size: int, processing_time: float) -> None:
        """Update processing metrics."""
        self.total_items_processed += batch_size
        self.total_batches_processed += 1
        
        # Update average batch size
        self.avg_batch_size = self.total_items_processed / self.total_batches_processed
        
        # Update average processing time
        if self.total_batches_processed == 1:
            self.avg_processing_time = processing_time
        else:
            self.avg_processing_time = ((self.avg_processing_time * (self.total_batches_processed - 1) + 
                                       processing_time) / self.total_batches_processed)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "queue_size": self.batch_queue.qsize(),
                "current_batch_size": len(self.current_batch),
                "total_items_processed": self.total_items_processed,
                "total_batches_processed": self.total_batches_processed,
                "avg_batch_size": round(self.avg_batch_size, 2),
                "avg_processing_time_ms": round(self.avg_processing_time * 1000, 2),
                "throughput_items_per_sec": round(self.total_items_processed / max(self.avg_processing_time * self.total_batches_processed, 0.001), 2) if self.total_batches_processed > 0 else 0,
                "config": {
                    "max_batch_size": self.config.max_batch_size,
                    "max_wait_time": self.config.max_wait_time,
                    "strategy": self.config.strategy.value,
                    "parallel_workers": self.config.parallel_workers
                }
            }


class NeuralBatchProcessor(BatchProcessor):
    """
    Specialized batch processor for neural data with domain-specific optimizations.
    """
    
    def __init__(self, config: BatchConfig = None):
        # Neural data specific configuration
        neural_config = config or BatchConfig(
            max_batch_size=50,  # Smaller batches for neural data
            max_wait_time=0.5,  # Faster processing for real-time requirements
            strategy=BatchStrategy.ADAPTIVE,
            parallel_workers=2   # Conservative for neural processing
        )
        
        super().__init__(self._process_neural_batch, neural_config)
        
        # Neural-specific metrics
        self.signal_quality_scores: List[float] = []
        self.processing_latencies: List[float] = []
        self.feature_extraction_times: List[float] = []
    
    def _process_neural_batch(self, batch: List[BatchItem]) -> Dict[str, Any]:
        """Process batch of neural data."""
        start_time = time.time()
        
        # Separate different types of neural data
        raw_data_items = []
        feature_items = []
        decode_items = []
        
        for item in batch:
            data_type = item.metadata.get('type', 'unknown')
            if data_type == 'raw_neural':
                raw_data_items.append(item)
            elif data_type == 'features':
                feature_items.append(item)
            elif data_type == 'decode_request':
                decode_items.append(item)
        
        results = {
            'processed_count': len(batch),
            'raw_data_processed': 0,
            'features_extracted': 0,
            'decoding_results': [],
            'batch_quality_score': 0.0
        }
        
        # Process raw neural data
        if raw_data_items:
            results['raw_data_processed'] = self._process_raw_neural_data(raw_data_items)
        
        # Process feature extraction
        if feature_items:
            results['features_extracted'] = self._process_feature_extraction(feature_items)
        
        # Process decoding requests
        if decode_items:
            results['decoding_results'] = self._process_decoding_requests(decode_items)
        
        # Calculate batch quality metrics
        if batch:
            quality_scores = [item.metadata.get('quality_score', 0.5) for item in batch]
            results['batch_quality_score'] = np.mean(quality_scores)
            self.signal_quality_scores.extend(quality_scores)
        
        processing_time = time.time() - start_time
        self.processing_latencies.append(processing_time)
        
        return results
    
    def _process_raw_neural_data(self, items: List[BatchItem]) -> int:
        """Process raw neural data items."""
        processed = 0
        
        for item in items:
            try:
                neural_data = item.data
                if isinstance(neural_data, np.ndarray):
                    # Basic preprocessing
                    # Apply bandpass filtering, artifact removal, etc.
                    processed_data = self._preprocess_neural_signal(neural_data)
                    
                    # Update item with processed data
                    item.data = processed_data
                    processed += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing raw neural data: {e}")
        
        return processed
    
    def _preprocess_neural_signal(self, data: np.ndarray) -> np.ndarray:
        """Basic neural signal preprocessing."""
        # Simple preprocessing pipeline
        # In production, this would include proper filtering
        
        # Remove DC offset
        data_centered = data - np.mean(data, axis=1, keepdims=True)
        
        # Simple outlier removal (clip to 3 standard deviations)
        std_dev = np.std(data_centered, axis=1, keepdims=True)
        data_clipped = np.clip(data_centered, -3 * std_dev, 3 * std_dev)
        
        return data_clipped
    
    def _process_feature_extraction(self, items: List[BatchItem]) -> int:
        """Process feature extraction requests."""
        start_time = time.time()
        processed = 0
        
        for item in items:
            try:
                neural_data = item.data
                paradigm = item.metadata.get('paradigm', 'P300')
                
                # Extract features based on paradigm
                features = self._extract_features(neural_data, paradigm)
                
                # Update item with extracted features
                item.data = features
                processed += 1
                
            except Exception as e:
                self.logger.error(f"Error extracting features: {e}")
        
        extraction_time = time.time() - start_time
        self.feature_extraction_times.append(extraction_time)
        
        return processed
    
    def _extract_features(self, data: np.ndarray, paradigm: str) -> np.ndarray:
        """Extract features from neural data based on paradigm."""
        if paradigm == 'P300':
            # P300 feature extraction (simplified)
            # In production, this would include proper ERP analysis
            features = np.mean(data, axis=1)  # Channel averages
        elif paradigm == 'MotorImagery':
            # Motor imagery features (simplified)
            features = np.std(data, axis=1)   # Channel variances
        elif paradigm == 'SSVEP':
            # SSVEP features (simplified)
            features = np.max(data, axis=1)   # Peak amplitudes
        else:
            # Default feature extraction
            features = np.concatenate([np.mean(data, axis=1), np.std(data, axis=1)])
        
        return features
    
    def _process_decoding_requests(self, items: List[BatchItem]) -> List[Dict[str, Any]]:
        """Process neural decoding requests."""
        results = []
        
        for item in items:
            try:
                features = item.data
                paradigm = item.metadata.get('paradigm', 'P300')
                
                # Simulate decoding (in production, use trained models)
                prediction = self._decode_neural_features(features, paradigm)
                
                result = {
                    'item_id': id(item),
                    'prediction': prediction,
                    'confidence': np.random.uniform(0.3, 0.9),  # Simulated confidence
                    'paradigm': paradigm
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error decoding neural data: {e}")
                results.append({
                    'item_id': id(item),
                    'error': str(e)
                })
        
        return results
    
    def _decode_neural_features(self, features: np.ndarray, paradigm: str) -> Any:
        """Decode neural features to intention."""
        # Simplified decoding simulation
        if paradigm == 'P300':
            return 1 if np.mean(features) > 0 else 0
        elif paradigm == 'MotorImagery':
            return np.argmax(features[:4]) if len(features) >= 4 else 0
        elif paradigm == 'SSVEP':
            return np.argmax(features[:4]) if len(features) >= 4 else 0
        else:
            return 0
    
    def add_neural_data(self, data: np.ndarray, data_type: str, paradigm: str = 'P300',
                       quality_score: float = 0.5, priority: int = 0) -> None:
        """Add neural data for batch processing."""
        metadata = {
            'type': data_type,
            'paradigm': paradigm,
            'quality_score': quality_score,
            'channels': data.shape[0] if len(data.shape) > 1 else 1,
            'samples': data.shape[-1]
        }
        
        self.add_item(data, priority, metadata)
    
    def get_neural_stats(self) -> Dict[str, Any]:
        """Get neural-specific processing statistics."""
        base_stats = self.get_stats()
        
        neural_stats = {
            'avg_signal_quality': round(np.mean(self.signal_quality_scores), 3) if self.signal_quality_scores else 0.0,
            'min_signal_quality': round(np.min(self.signal_quality_scores), 3) if self.signal_quality_scores else 0.0,
            'max_signal_quality': round(np.max(self.signal_quality_scores), 3) if self.signal_quality_scores else 0.0,
            'avg_feature_extraction_time_ms': round(np.mean(self.feature_extraction_times) * 1000, 2) if self.feature_extraction_times else 0.0,
            'processing_latency_p95_ms': round(np.percentile(self.processing_latencies, 95) * 1000, 2) if self.processing_latencies else 0.0,
            'total_neural_samples': len(self.signal_quality_scores)
        }
        
        base_stats.update(neural_stats)
        return base_stats


# Factory functions
def create_batch_processor(processor_type: str, **kwargs) -> BatchProcessor:
    """Create batch processor of specified type."""
    
    if processor_type == "neural":
        return NeuralBatchProcessor(**kwargs)
    elif processor_type == "generic":
        if 'processor_func' not in kwargs:
            raise ValueError("Generic batch processor requires processor_func")
        return BatchProcessor(**kwargs)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


# Example usage
if __name__ == "__main__":
    import time
    import random
    
    # Test neural batch processor
    neural_processor = NeuralBatchProcessor()
    neural_processor.start()
    
    try:
        # Add some test neural data
        for i in range(20):
            test_data = np.random.randn(8, 250)  # 8 channels, 250 samples
            neural_processor.add_neural_data(
                test_data, 
                'raw_neural', 
                'P300', 
                quality_score=random.uniform(0.3, 0.9),
                priority=random.randint(0, 2)
            )
            time.sleep(0.1)
        
        # Wait for processing
        time.sleep(3)
        
        # Print statistics
        stats = neural_processor.get_neural_stats()
        print("Neural Batch Processor Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    finally:
        neural_processor.stop()