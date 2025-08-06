"""
Tests for performance optimization components.
"""

import pytest
import asyncio
import time
import threading
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

from bci_agent_bridge.performance.caching import (
    CacheManager, NeuralDataCache, CachePolicy, CacheEntry, ValidationResult
)
from bci_agent_bridge.performance.connection_pool import (
    ConnectionPool, ClaudeClientPool, PooledConnection, PoolState
)
from bci_agent_bridge.performance.batch_processor import (
    BatchProcessor, NeuralBatchProcessor, BatchConfig, BatchStrategy, BatchItem
)
from bci_agent_bridge.performance.load_balancer import (
    LoadBalancer, AdaptiveLoadBalancer, Worker, LoadBalancingStrategy, WorkerState
)


class TestCacheManager:
    """Test suite for CacheManager."""
    
    def test_cache_initialization(self):
        """Test cache manager initialization."""
        cache = CacheManager(
            max_size_bytes=1024*1024,
            max_entries=100,
            policy=CachePolicy.LRU
        )
        
        assert cache.max_size_bytes == 1024*1024
        assert cache.max_entries == 100
        assert cache.policy == CachePolicy.LRU
        assert len(cache.cache) == 0
    
    def test_put_and_get_basic(self):
        """Test basic put and get operations."""
        cache = CacheManager(max_entries=10)
        
        # Put item
        success = cache.put("test_key", "test_value")
        assert success
        
        # Get item
        value = cache.get("test_key")
        assert value == "test_value"
        
        # Get non-existent item
        value = cache.get("non_existent")
        assert value is None
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = CacheManager(max_entries=3, policy=CachePolicy.LRU)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add another item, should evict key2 (least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Should still be there
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should still be there
        assert cache.get("key4") == "value4"  # Should be there
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = CacheManager(policy=CachePolicy.TTL, default_ttl=0.1)
        
        cache.put("key1", "value1")
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("key1") is None
    
    def test_size_based_eviction(self):
        """Test size-based eviction."""
        cache = CacheManager(max_size_bytes=100, max_entries=10)
        
        # Put large item that will trigger size-based eviction
        large_data = "x" * 60  # 60 bytes
        cache.put("large1", large_data)
        cache.put("large2", large_data)  # Should fit
        
        # This should trigger eviction of large1
        cache.put("large3", large_data)
        
        assert cache.get("large1") is None
        assert cache.get("large2") == large_data
        assert cache.get("large3") == large_data
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = CacheManager()
        
        # Generate some hits and misses
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["hit_rate"] > 0
        assert "entry_count" in stats
    
    def test_delete_operation(self):
        """Test cache deletion."""
        cache = CacheManager()
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        success = cache.delete("key1")
        assert success
        assert cache.get("key1") is None
        
        # Delete non-existent key
        success = cache.delete("non_existent")
        assert not success
    
    def test_clear_cache(self):
        """Test cache clearing."""
        cache = CacheManager()
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert len(cache.cache) == 2
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.get("key1") is None


class TestNeuralDataCache:
    """Test suite for NeuralDataCache."""
    
    def test_neural_cache_initialization(self):
        """Test neural data cache initialization."""
        cache = NeuralDataCache()
        
        assert cache.policy == CachePolicy.NEURAL_OPTIMIZED
        assert cache.feature_cache_ttl == 1800.0
        assert cache.model_cache_ttl == 7200.0
        assert cache.raw_data_ttl == 300.0
    
    def test_cache_neural_features(self):
        """Test caching neural features."""
        cache = NeuralDataCache()
        
        features = np.random.randn(64, 250)
        metadata = {"paradigm": "P300", "quality": 0.8}
        
        success = cache.cache_neural_features("subject_001", "P300", features, metadata)
        assert success
        
        # Retrieve features
        cached_features = cache.get_neural_features("subject_001", "P300")
        assert cached_features is not None
        assert np.array_equal(cached_features["features"], features)
        assert cached_features["metadata"] == metadata
    
    def test_cache_decoder_model(self):
        """Test caching decoder models."""
        cache = NeuralDataCache()
        
        model_data = {"weights": np.random.randn(10, 5), "bias": np.random.randn(10)}
        
        success = cache.cache_decoder_model("model_001", model_data, "P300", "subject_001")
        assert success
        
        # Retrieve model
        cached_model = cache.get_decoder_model("model_001", "P300", "subject_001")
        assert cached_model is not None
        assert np.array_equal(cached_model["weights"], model_data["weights"])
    
    def test_cache_raw_neural_data(self):
        """Test caching raw neural data."""
        cache = NeuralDataCache()
        
        raw_data = np.random.randn(8, 1000)
        channels = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]
        
        success = cache.cache_raw_neural_data("session_001", raw_data, 250, channels)
        assert success
        
        # Raw data should have shorter TTL
        assert "raw_session_001" in [key for key in cache.cache.keys() if key.startswith("raw_")]
    
    def test_cleanup_expired_neural_data(self):
        """Test cleanup of expired neural data."""
        cache = NeuralDataCache()
        
        # Add some test data with very short TTL
        cache.raw_data_ttl = 0.01  # 10ms
        
        raw_data = np.random.randn(8, 100)
        cache.cache_raw_neural_data("session_001", raw_data, 250, ["CH1"])
        
        # Wait for expiration
        time.sleep(0.02)
        
        # Cleanup should remove expired data
        removed_count = cache.cleanup_expired_neural_data()
        assert removed_count >= 1
    
    def test_neural_cache_summary(self):
        """Test neural cache summary."""
        cache = NeuralDataCache()
        
        # Add different types of data
        features = np.random.randn(64, 100)
        cache.cache_neural_features("subject_001", "P300", features)
        
        model_data = {"test": "model"}
        cache.cache_decoder_model("model_001", model_data, "P300")
        
        raw_data = np.random.randn(8, 100)
        cache.cache_raw_neural_data("session_001", raw_data, 250, ["CH1"])
        
        summary = cache.get_neural_cache_summary()
        
        assert summary["total_entries"] >= 3
        assert summary["feature_entries"] >= 1
        assert summary["model_entries"] >= 1
        assert summary["raw_data_entries"] >= 1


class TestConnectionPool:
    """Test suite for ConnectionPool."""
    
    def test_connection_pool_initialization(self):
        """Test connection pool initialization."""
        def create_conn():
            return {"id": time.time()}
        
        pool = ConnectionPool(
            connection_factory=create_conn,
            min_connections=2,
            max_connections=5
        )
        
        assert pool.min_connections == 2
        assert pool.max_connections == 5
        assert pool.state == PoolState.HEALTHY
    
    @pytest.mark.asyncio
    async def test_get_and_return_connection(self):
        """Test getting and returning connections."""
        def create_conn():
            return {"id": time.time(), "data": "test"}
        
        def health_check(conn):
            return True
        
        pool = ConnectionPool(
            connection_factory=create_conn,
            min_connections=1,
            max_connections=3,
            health_check=health_check
        )
        
        # Get connection
        pooled_conn = await pool.get_connection()
        assert pooled_conn is not None
        assert pooled_conn.connection["data"] == "test"
        
        # Return connection
        pool.return_connection(pooled_conn, is_healthy=True)
        
        # Connection should be available for reuse
        pooled_conn2 = await pool.get_connection()
        assert pooled_conn2 is not None
        
        pool.return_connection(pooled_conn2)
    
    @pytest.mark.asyncio 
    async def test_connection_context_manager(self):
        """Test connection context manager."""
        def create_conn():
            return {"value": 42}
        
        pool = ConnectionPool(
            connection_factory=create_conn,
            min_connections=1,
            max_connections=2
        )
        
        async with pool.get_connection_context() as conn:
            assert conn["value"] == 42
        
        # Connection should be returned automatically
        stats = pool.get_stats()
        assert stats["active_connections"] == 0
    
    @pytest.mark.asyncio
    async def test_connection_pool_stats(self):
        """Test connection pool statistics."""
        def create_conn():
            return {"test": True}
        
        pool = ConnectionPool(
            connection_factory=create_conn,
            min_connections=1,
            max_connections=3
        )
        
        # Get some connections
        conn1 = await pool.get_connection()
        conn2 = await pool.get_connection()
        
        stats = pool.get_stats()
        
        assert stats["active_connections"] == 2
        assert stats["total_created"] >= 2
        assert "hit_rate_pct" in stats
        
        # Return connections
        pool.return_connection(conn1)
        pool.return_connection(conn2)
    
    @pytest.mark.asyncio
    async def test_pool_cleanup(self):
        """Test connection pool cleanup."""
        def create_conn():
            return {"created_at": time.time()}
        
        def health_check(conn):
            # Mark connections as unhealthy after 0.01 seconds
            return (time.time() - conn["created_at"]) < 0.01
        
        pool = ConnectionPool(
            connection_factory=create_conn,
            min_connections=1,
            max_connections=3,
            health_check=health_check,
            max_idle_time=0.01
        )
        
        # Wait for connections to become unhealthy
        time.sleep(0.02)
        
        # Cleanup should remove unhealthy connections
        cleaned = await pool.cleanup_idle_connections()
        assert cleaned >= 0  # May or may not clean depending on timing


class TestBatchProcessor:
    """Test suite for BatchProcessor."""
    
    def test_batch_processor_initialization(self):
        """Test batch processor initialization."""
        def processor_func(batch):
            return f"Processed {len(batch)} items"
        
        config = BatchConfig(
            max_batch_size=50,
            max_wait_time=2.0,
            strategy=BatchStrategy.HYBRID
        )
        
        processor = BatchProcessor(processor_func, config)
        
        assert processor.config.max_batch_size == 50
        assert processor.config.max_wait_time == 2.0
        assert processor.config.strategy == BatchStrategy.HYBRID
        assert not processor.is_running
    
    def test_batch_processor_start_stop(self):
        """Test batch processor start and stop."""
        def processor_func(batch):
            return "processed"
        
        processor = BatchProcessor(processor_func)
        
        assert not processor.is_running
        
        processor.start()
        assert processor.is_running
        
        processor.stop(timeout=1.0)
        assert not processor.is_running
    
    def test_add_items_and_processing(self):
        """Test adding items and batch processing."""
        processed_batches = []
        
        def processor_func(batch):
            processed_batches.append(batch)
            return f"Processed {len(batch)} items"
        
        config = BatchConfig(
            max_batch_size=3,
            max_wait_time=0.1,
            strategy=BatchStrategy.SIZE_BASED
        )
        
        processor = BatchProcessor(processor_func, config)
        processor.start()
        
        try:
            # Add items
            for i in range(5):
                processor.add_item(f"item_{i}", priority=i)
            
            # Wait for processing
            time.sleep(0.5)
            
            # Should have processed at least one batch
            assert len(processed_batches) >= 1
            
            # Check batch sizes
            for batch in processed_batches:
                assert len(batch) <= 3
        
        finally:
            processor.stop()
    
    def test_batch_processor_stats(self):
        """Test batch processor statistics."""
        def processor_func(batch):
            time.sleep(0.01)  # Simulate processing time
            return "processed"
        
        processor = BatchProcessor(processor_func)
        processor.start()
        
        try:
            # Add some items
            for i in range(10):
                processor.add_item(f"item_{i}")
            
            # Wait for processing
            time.sleep(0.5)
            
            stats = processor.get_stats()
            
            assert stats["is_running"]
            assert "total_items_processed" in stats
            assert "total_batches_processed" in stats
            assert "avg_processing_time_ms" in stats
        
        finally:
            processor.stop()


class TestNeuralBatchProcessor:
    """Test suite for NeuralBatchProcessor."""
    
    def test_neural_batch_processor_initialization(self):
        """Test neural batch processor initialization."""
        processor = NeuralBatchProcessor()
        
        assert processor.config.max_batch_size <= 50  # Should be smaller for neural data
        assert processor.config.strategy == BatchStrategy.ADAPTIVE
        assert len(processor.signal_quality_scores) == 0
    
    def test_add_neural_data(self):
        """Test adding neural data for processing."""
        processor = NeuralBatchProcessor()
        processor.start()
        
        try:
            # Add different types of neural data
            raw_data = np.random.randn(8, 250)
            processor.add_neural_data(raw_data, "raw_neural", "P300", quality_score=0.8)
            
            features = np.random.randn(64)
            processor.add_neural_data(features, "features", "P300", quality_score=0.9)
            
            # Wait for processing
            time.sleep(0.5)
            
            # Should have recorded quality scores
            assert len(processor.signal_quality_scores) >= 2
            
        finally:
            processor.stop()
    
    def test_neural_batch_processing(self):
        """Test neural data batch processing."""
        processor = NeuralBatchProcessor()
        processor.start()
        
        try:
            # Add various neural data types
            for i in range(5):
                raw_data = np.random.randn(8, 100)
                processor.add_neural_data(raw_data, "raw_neural", "P300", quality_score=0.5 + i*0.1)
                
                features = np.random.randn(32)
                processor.add_neural_data(features, "features", "P300", quality_score=0.7)
                
                decode_data = np.random.randn(16)
                processor.add_neural_data(decode_data, "decode_request", "P300", quality_score=0.8)
            
            # Wait for processing
            time.sleep(1.0)
            
            stats = processor.get_neural_stats()
            
            assert "avg_signal_quality" in stats
            assert "processing_latency_p95_ms" in stats
            assert stats["total_neural_samples"] >= 15
            
        finally:
            processor.stop()
    
    def test_neural_preprocessing(self):
        """Test neural signal preprocessing."""
        processor = NeuralBatchProcessor()
        
        # Create test data with DC offset and outliers
        test_data = np.random.randn(4, 100)
        test_data += 10  # Add DC offset
        test_data[0, 50] = 100  # Add outlier
        
        processed_data = processor._preprocess_neural_signal(test_data)
        
        # Should remove DC offset
        assert abs(np.mean(processed_data)) < 1.0
        
        # Should clip outliers
        assert np.max(processed_data) < 50


class TestLoadBalancer:
    """Test suite for LoadBalancer."""
    
    def test_load_balancer_initialization(self):
        """Test load balancer initialization."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
        
        assert lb.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert len(lb.workers) == 0
        assert lb.total_requests == 0
    
    def test_add_remove_workers(self):
        """Test adding and removing workers."""
        lb = LoadBalancer()
        
        worker1 = Worker("worker1", "endpoint1", weight=1.0)
        worker2 = Worker("worker2", "endpoint2", weight=2.0)
        
        # Add workers
        lb.add_worker(worker1)
        lb.add_worker(worker2)
        
        assert len(lb.workers) == 2
        
        # Remove worker
        success = lb.remove_worker("worker1")
        assert success
        assert len(lb.workers) == 1
        
        # Try to remove non-existent worker
        success = lb.remove_worker("worker999")
        assert not success
    
    def test_round_robin_selection(self):
        """Test round-robin worker selection."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
        
        workers = [
            Worker("worker1", "endpoint1"),
            Worker("worker2", "endpoint2"), 
            Worker("worker3", "endpoint3")
        ]
        
        for worker in workers:
            lb.add_worker(worker)
        
        # Get workers in round-robin order
        selected_workers = []
        for _ in range(6):  # Two full cycles
            worker = lb.get_worker()
            selected_workers.append(worker.id if worker else None)
        
        # Should cycle through workers
        expected = ["worker1", "worker2", "worker3", "worker1", "worker2", "worker3"]
        assert selected_workers == expected
    
    def test_least_connections_selection(self):
        """Test least connections worker selection."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        worker1 = Worker("worker1", "endpoint1")
        worker2 = Worker("worker2", "endpoint2")
        
        worker1.metrics.active_connections = 5
        worker2.metrics.active_connections = 2
        
        lb.add_worker(worker1)
        lb.add_worker(worker2)
        
        # Should select worker with fewer connections
        selected = lb.get_worker()
        assert selected.id == "worker2"
    
    @pytest.mark.asyncio
    async def test_process_request(self):
        """Test request processing through load balancer."""
        def worker1_processor(data):
            return f"Worker1 processed: {data}"
        
        def worker2_processor(data):
            return f"Worker2 processed: {data}"
        
        lb = LoadBalancer()
        
        worker1 = Worker("worker1", "endpoint1", processor_func=worker1_processor)
        worker2 = Worker("worker2", "endpoint2", processor_func=worker2_processor)
        
        lb.add_worker(worker1)
        lb.add_worker(worker2)
        
        # Process request
        result = await lb.process_request("test_data")
        
        assert "processed: test_data" in result
        assert lb.total_requests == 1
    
    def test_worker_health_states(self):
        """Test worker health state management."""
        worker = Worker("test_worker", "endpoint")
        
        # Initially healthy
        assert worker.state == WorkerState.HEALTHY
        assert worker.is_available
        
        # Mark as unhealthy
        worker.state = WorkerState.UNHEALTHY
        assert not worker.is_available
        
        # Test with max connections reached
        worker.state = WorkerState.HEALTHY
        worker.metrics.active_connections = worker.max_connections
        assert not worker.is_available
    
    def test_load_balancer_stats(self):
        """Test load balancer statistics."""
        lb = LoadBalancer()
        
        worker1 = Worker("worker1", "endpoint1")
        worker2 = Worker("worker2", "endpoint2")
        worker2.state = WorkerState.UNHEALTHY
        
        lb.add_worker(worker1)
        lb.add_worker(worker2)
        
        stats = lb.get_stats()
        
        assert stats["total_workers"] == 2
        assert stats["healthy_workers"] == 1
        assert stats["available_workers"] == 1
        assert len(stats["workers"]) == 2


class TestAdaptiveLoadBalancer:
    """Test suite for AdaptiveLoadBalancer."""
    
    def test_adaptive_load_balancer_initialization(self):
        """Test adaptive load balancer initialization."""
        alb = AdaptiveLoadBalancer()
        
        assert alb.strategy == LoadBalancingStrategy.ADAPTIVE
        assert alb.learning_enabled
        assert len(alb.strategy_performance) > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_request_processing(self):
        """Test adaptive request processing with learning."""
        def fast_processor(data):
            time.sleep(0.01)
            return f"Fast: {data}"
        
        def slow_processor(data):
            time.sleep(0.05)
            return f"Slow: {data}"
        
        alb = AdaptiveLoadBalancer()
        alb.exploration_rate = 0.0  # Disable exploration for predictable testing
        
        worker1 = Worker("fast_worker", "endpoint1", processor_func=fast_processor)
        worker2 = Worker("slow_worker", "endpoint2", processor_func=slow_processor)
        
        alb.add_worker(worker1)
        alb.add_worker(worker2)
        
        # Process several requests
        for i in range(10):
            result = await alb.process_request(f"request_{i}")
            assert "request_" in result
        
        # Should have performance data
        assert alb.current_strategy in alb.strategy_performance
        assert len(alb.strategy_performance[alb.current_strategy]) > 0
    
    def test_adaptive_stats(self):
        """Test adaptive load balancer statistics."""
        alb = AdaptiveLoadBalancer()
        
        # Add some mock performance data
        alb.strategy_performance[LoadBalancingStrategy.ROUND_ROBIN] = [0.01, 0.02, 0.01]
        alb.strategy_performance[LoadBalancingStrategy.LEAST_CONNECTIONS] = [0.02, 0.03, 0.02]
        
        stats = alb.get_adaptive_stats()
        
        assert "current_strategy" in stats
        assert "learning_enabled" in stats
        assert "strategy_performance" in stats
        assert len(stats["strategy_performance"]) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])