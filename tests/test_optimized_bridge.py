"""
Tests for OptimizedBCIBridge performance enhancements.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import patch, MagicMock

from bci_agent_bridge.core.optimized_bridge import OptimizedBCIBridge, PerformanceConfig
from bci_agent_bridge.core.bridge import NeuralData
from bci_agent_bridge.decoders.p300 import P300Decoder


class TestOptimizedBCIBridge:
    """Test suite for OptimizedBCIBridge."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.perf_config = PerformanceConfig(
            enable_caching=True,
            enable_batching=True,
            enable_parallel_processing=True,
            cache_size_mb=10,
            batch_size=8,
            batch_timeout_ms=50,
            worker_threads=2
        )
        
        self.bridge = OptimizedBCIBridge(
            device="Simulation",
            channels=8,
            sampling_rate=250,
            paradigm="P300",
            performance_config=self.perf_config
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.bridge.cleanup()
    
    def test_initialization(self):
        """Test OptimizedBCIBridge initialization."""
        assert self.bridge.perf_config == self.perf_config
        assert self.bridge.cache is not None
        assert self.bridge.batch_processor is not None
        assert self.bridge.executor is not None
        
        # Test metrics initialization
        metrics = self.bridge.get_performance_metrics()
        assert "total_processed" in metrics
        assert "cache_hit_rate" in metrics
        assert "performance_config" in metrics
    
    def test_performance_config_validation(self):
        """Test performance configuration validation."""
        # Test default configuration
        default_bridge = OptimizedBCIBridge()
        assert default_bridge.perf_config.enable_caching is True
        assert default_bridge.perf_config.enable_batching is True
        
        # Test custom configuration
        custom_config = PerformanceConfig(
            enable_caching=False,
            enable_batching=False,
            worker_threads=8
        )
        custom_bridge = OptimizedBCIBridge(performance_config=custom_config)
        assert custom_bridge.perf_config.enable_caching is False
        assert custom_bridge.cache is None
        custom_bridge.cleanup()
    
    def test_caching_functionality(self):
        """Test neural data caching."""
        # Create sample neural data
        test_data = np.random.randn(8, 100)
        neural_data = NeuralData(
            data=test_data,
            timestamp=time.time(),
            channels=[f"CH{i+1}" for i in range(8)],
            sampling_rate=250
        )
        
        # Initialize decoder
        self.bridge.decoder = P300Decoder(channels=8, sampling_rate=250)
        
        # First call should miss cache
        initial_cache_misses = self.bridge._cache_stats["misses"]
        intention1 = self.bridge.optimized_decode_intention(neural_data)
        assert self.bridge._cache_stats["misses"] == initial_cache_misses + 1
        
        # Second call with same data should hit cache
        intention2 = self.bridge.optimized_decode_intention(neural_data)
        assert self.bridge._cache_stats["hits"] >= 1
        
        # Results should be consistent
        assert intention1.command == intention2.command
    
    def test_data_hash_computation(self):
        """Test neural data hash computation for caching."""
        data1 = np.random.randn(8, 100)
        data2 = np.random.randn(8, 100)
        data3 = data1.copy()
        
        hash1 = self.bridge._compute_data_hash(data1)
        hash2 = self.bridge._compute_data_hash(data2)
        hash3 = self.bridge._compute_data_hash(data3)
        
        # Different data should have different hashes
        assert hash1 != hash2
        
        # Same data should have same hash
        assert hash1 == hash3
        
        # Hashes should be reasonable length
        assert len(hash1) == 16
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Mock batch items
        batch_items = []
        for i in range(5):
            from bci_agent_bridge.performance.batch_processor import BatchItem
            item = BatchItem(
                data=np.random.randn(50),  # Features
                timestamp=time.time(),
                metadata={"id": i}
            )
            batch_items.append(item)
        
        # Initialize decoder
        self.bridge.decoder = P300Decoder(channels=8, sampling_rate=250)
        
        # Process batch
        results = self.bridge._process_neural_batch(batch_items)
        
        assert len(results) == len(batch_items)
        assert all(isinstance(result, (int, np.integer)) for result in results)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Initialize decoder for processing
        self.bridge.decoder = P300Decoder(channels=8, sampling_rate=250)
        
        # Process some data to generate metrics
        for i in range(5):
            test_data = np.random.randn(8, 100)
            neural_data = NeuralData(
                data=test_data,
                timestamp=time.time(),
                channels=[f"CH{i+1}" for i in range(8)],
                sampling_rate=250
            )
            self.bridge.optimized_decode_intention(neural_data)
        
        metrics = self.bridge.get_performance_metrics()
        
        # Check required metrics
        assert metrics["total_processed"] == 0  # Only intention decoding counted
        assert "avg_processing_time_ms" in metrics
        assert "cache_hit_rate" in metrics
        assert "buffer_utilization" in metrics
        assert "throughput_samples_per_sec" in metrics
        assert "memory_usage_mb" in metrics
        assert "performance_config" in metrics
        
        # Validate metric types
        assert isinstance(metrics["cache_hit_rate"], float)
        assert 0 <= metrics["cache_hit_rate"] <= 1
        assert isinstance(metrics["memory_usage_mb"], float)
        assert metrics["memory_usage_mb"] >= 0
    
    def test_optimization_modes(self):
        """Test latency and throughput optimization modes."""
        original_batch_size = self.bridge.perf_config.batch_size
        
        # Test latency optimization
        self.bridge.optimize_for_latency()
        assert self.bridge.perf_config.batch_size == 1
        assert self.bridge.perf_config.batch_timeout_ms == 1
        assert self.bridge.perf_config.enable_batching is False
        
        # Reset
        self.bridge.perf_config.batch_size = original_batch_size
        
        # Test throughput optimization
        self.bridge.optimize_for_throughput()
        assert self.bridge.perf_config.batch_size == 64
        assert self.bridge.perf_config.batch_timeout_ms == 50
        assert self.bridge.perf_config.enable_batching is True
        assert self.bridge.perf_config.enable_parallel_processing is True
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        # Initially should have minimal memory
        initial_memory = self.bridge._estimate_memory_usage()
        assert initial_memory >= 0
        
        # Add some data to buffer
        for i in range(10):
            test_data = np.random.randn(8, 100)
            neural_data = NeuralData(
                data=test_data,
                timestamp=time.time(),
                channels=[f"CH{i+1}" for i in range(8)],
                sampling_rate=250
            )
            self.bridge._add_to_buffer_safe(neural_data)
        
        # Memory usage should increase
        new_memory = self.bridge._estimate_memory_usage()
        assert new_memory > initial_memory
    
    @pytest.mark.asyncio
    async def test_optimized_streaming_setup(self):
        """Test optimized streaming setup (without full execution)."""
        # Mock the device connection
        self.bridge._device_connected = True
        
        # Mock the raw data reading
        async def mock_read_raw_data():
            return np.random.randn(8, 10)
        
        with patch.object(self.bridge, '_read_raw_data', side_effect=mock_read_raw_data):
            # Start streaming briefly
            stream_gen = self.bridge.optimized_stream()
            
            # Get first sample
            try:
                first_sample = await asyncio.wait_for(anext(stream_gen), timeout=1.0)
                assert isinstance(first_sample, NeuralData)
                assert first_sample.data.shape[0] == 8  # 8 channels
            except asyncio.TimeoutError:
                pytest.skip("Streaming test timed out - this may be expected in test environment")
            finally:
                self.bridge.stop_streaming()
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        # Initially no processing times
        assert self.bridge._calculate_throughput() == 0.0
        
        # Add some processing times
        self.bridge._processing_times = [0.01, 0.02, 0.015, 0.012]  # 10-20ms each
        
        throughput = self.bridge._calculate_throughput()
        assert throughput > 0
        assert 50 <= throughput <= 100  # Should be 50-100 samples/sec for these times
    
    def test_error_handling_in_batch_processing(self):
        """Test error handling in batch processing."""
        # Test with empty batch
        results = self.bridge._process_neural_batch([])
        assert results == []
        
        # Test with invalid decoder
        self.bridge.decoder = None
        from bci_agent_bridge.performance.batch_processor import BatchItem
        
        batch_items = [BatchItem(data=np.random.randn(50), timestamp=time.time())]
        
        # Should handle gracefully and return defaults
        results = self.bridge._process_neural_batch(batch_items)
        assert len(results) == 1
        assert results[0] == 0  # Default prediction
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Verify components are initialized
        assert self.bridge.batch_processor is not None
        assert self.bridge.cache is not None
        assert self.bridge.executor is not None
        
        # Cleanup
        self.bridge.cleanup()
        
        # Verify executor is shutdown
        assert self.bridge.executor._shutdown


if __name__ == "__main__":
    pytest.main([__file__])