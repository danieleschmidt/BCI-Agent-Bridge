"""
Integration tests for BCI-Agent-Bridge components.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import json

from bci_agent_bridge.core.bridge import BCIBridge, NeuralData, DecodedIntention
from bci_agent_bridge.adapters.claude_flow import ClaudeFlowAdapter, ClaudeResponse
from bci_agent_bridge.monitoring.health_monitor import HealthMonitor, create_bci_health_checks
from bci_agent_bridge.monitoring.metrics_collector import BCIMetricsCollector
from bci_agent_bridge.monitoring.alert_manager import AlertManager, create_bci_alert_rules
from bci_agent_bridge.performance.caching import NeuralDataCache
from bci_agent_bridge.performance.batch_processor import NeuralBatchProcessor
from bci_agent_bridge.performance.connection_pool import ConnectionPool
from bci_agent_bridge.performance.load_balancer import LoadBalancer, Worker


@pytest.mark.integration
class TestBCIBridgeIntegration:
    """Integration tests for BCI bridge with components."""
    
    def test_bridge_with_neural_cache_integration(self, mock_neural_data):
        """Test BCI bridge integration with neural data cache."""
        # Initialize components
        bridge = BCIBridge(device="Simulation", paradigm="P300")
        cache = NeuralDataCache()
        
        # Generate test data
        test_data = mock_neural_data(paradigm="P300")
        
        # Cache neural features
        features = np.mean(test_data, axis=1)  # Simple feature extraction
        success = cache.cache_neural_features("subject_001", "P300", features)
        assert success
        
        # Retrieve cached features
        cached_features = cache.get_neural_features("subject_001", "P300")
        assert cached_features is not None
        assert np.array_equal(cached_features["features"], features)
        
        # Test with decoder model caching
        model_data = {"weights": np.random.randn(8, 4), "bias": np.random.randn(4)}
        cache.cache_decoder_model("model_001", model_data, "P300", "subject_001")
        
        cached_model = cache.get_decoder_model("model_001", "P300", "subject_001")
        assert cached_model is not None
        assert np.array_equal(cached_model["weights"], model_data["weights"])
    
    def test_bridge_with_batch_processor_integration(self, mock_neural_data):
        """Test BCI bridge integration with batch processor."""
        bridge = BCIBridge(device="Simulation", paradigm="P300")
        processor = NeuralBatchProcessor()
        
        processor.start()
        
        try:
            # Add neural data to batch processor
            for i in range(5):
                neural_data = mock_neural_data(paradigm="P300")
                processor.add_neural_data(neural_data, "raw_neural", "P300", quality_score=0.8)
                
                # Add features
                features = np.mean(neural_data, axis=1)
                processor.add_neural_data(features, "features", "P300", quality_score=0.9)
                
                # Add decode requests
                processor.add_neural_data(features, "decode_request", "P300", quality_score=0.85)
            
            # Wait for processing
            time.sleep(1.0)
            
            stats = processor.get_neural_stats()
            assert stats["total_neural_samples"] >= 15
            assert stats["avg_signal_quality"] > 0
            
        finally:
            processor.stop()
    
    @pytest.mark.asyncio
    async def test_bridge_with_health_monitoring_integration(self, mock_bci_bridge):
        """Test BCI bridge integration with health monitoring."""
        monitor = HealthMonitor(check_interval=0.1)
        
        # Register BCI health checks
        health_checks = create_bci_health_checks(mock_bci_bridge)
        for name, check_func in health_checks.items():
            monitor.register_health_check(name, check_func)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        try:
            # Wait for a few health check cycles
            await asyncio.sleep(0.3)
            
            # Check results
            summary = monitor.get_health_summary()
            assert summary["total_active"] or summary["monitoring_active"]
            assert "components" in summary
            
        finally:
            await monitor.stop_monitoring()


@pytest.mark.integration 
class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    @pytest.mark.asyncio
    async def test_health_monitor_with_metrics_integration(self, mock_bci_bridge):
        """Test health monitor integration with metrics collector."""
        monitor = HealthMonitor(check_interval=0.1)
        metrics = BCIMetricsCollector()
        
        # Register health checks
        health_checks = create_bci_health_checks(mock_bci_bridge)
        for name, check_func in health_checks.items():
            monitor.register_health_check(name, check_func)
        
        # Custom health check that records metrics
        def metrics_recording_check():
            from bci_agent_bridge.monitoring.health_monitor import HealthCheck, HealthStatus
            
            # Record some metrics
            metrics.record_metric("health_check_duration", 10.0)
            metrics.increment_counter("health_checks_performed")
            
            return HealthCheck(
                name="metrics_check",
                status=HealthStatus.HEALTHY,
                message="Metrics recorded successfully",
                last_check=time.time(),
                duration_ms=10.0
            )
        
        monitor.register_health_check("metrics_recording", metrics_recording_check)
        
        await monitor.start_monitoring()
        
        try:
            # Wait for monitoring cycles
            await asyncio.sleep(0.3)
            
            # Check that metrics were recorded
            assert metrics.counters["health_checks_performed"] > 0
            assert "health_check_duration" in metrics.metrics
            
        finally:
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_metrics_with_alerts_integration(self):
        """Test metrics collector integration with alert manager."""
        metrics = BCIMetricsCollector()
        alert_manager = AlertManager()
        
        # Register BCI alert rules
        rules = create_bci_alert_rules()
        for rule in rules:
            alert_manager.register_alert_rule(rule)
        
        # Register notification handler
        triggered_alerts = []
        def test_handler(alert):
            triggered_alerts.append(alert)
        
        alert_manager.register_notification_handler("test", test_handler)
        
        # Set escalation policy
        from bci_agent_bridge.monitoring.alert_manager import AlertSeverity
        alert_manager.escalation_policies[AlertSeverity.WARNING] = ["test"]
        alert_manager.escalation_policies[AlertSeverity.CRITICAL] = ["test"]
        
        # Record metrics that should trigger alerts
        metrics.record_neural_sample(8, 40, 0.1)  # Low data rate + low quality
        metrics.record_decoding_event("P300", 0.3, 300.0)  # Low confidence + high latency
        metrics.record_system_performance(90.0, 2000.0, 1000, 2)  # High CPU + memory
        
        # Create context from metrics
        context = {
            "signal_quality": 0.1,
            "data_rate": 40,
            "avg_confidence": 0.3,
            "decoding_latency": 300.0,
            "cpu_usage": 90.0,
            "memory_usage": 2000.0
        }
        
        # Evaluate rules
        alerts = await alert_manager.evaluate_rules(context)
        
        assert len(alerts) >= 3  # Should trigger multiple alerts
        assert len(triggered_alerts) >= 3  # Should have notified
    
    def test_full_monitoring_stack_integration(self, mock_bci_bridge):
        """Test full monitoring stack integration."""
        # Initialize all monitoring components
        monitor = HealthMonitor(check_interval=0.1)
        metrics = BCIMetricsCollector()
        alert_manager = AlertManager()
        
        # Setup health checks
        health_checks = create_bci_health_checks(mock_bci_bridge)
        for name, check_func in health_checks.items():
            monitor.register_health_check(name, check_func)
        
        # Setup alert rules
        rules = create_bci_alert_rules()
        for rule in rules:
            alert_manager.register_alert_rule(rule)
        
        # Record various metrics
        metrics.record_neural_sample(8, 250, 0.8)
        metrics.record_decoding_event("P300", 0.9, 50.0)
        metrics.record_claude_interaction("medical", 100, 800.0)
        metrics.record_system_performance(25.0, 512.0, 10, 1)
        
        # Get comprehensive summary
        monitor_summary = monitor.get_health_summary()
        metrics_summary = metrics.get_bci_performance_summary()
        alert_summary = alert_manager.get_alert_summary()
        
        # Verify all components are working
        assert monitor_summary["monitoring_active"] or len(monitor_summary["components"]) > 0
        assert metrics_summary["neural_processing"]["samples_processed"] > 0
        assert alert_summary["registered_rules"] > 0


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for performance components."""
    
    def test_cache_with_batch_processor_integration(self, mock_neural_data):
        """Test cache integration with batch processor."""
        cache = NeuralDataCache()
        processor = NeuralBatchProcessor()
        
        processor.start()
        
        try:
            # Process some neural data through batch processor
            for i in range(3):
                neural_data = mock_neural_data(channels=8, samples=250)
                processor.add_neural_data(neural_data, "raw_neural", "P300")
            
            # Wait for processing
            time.sleep(0.5)
            
            # Cache processed results
            for i in range(3):
                features = np.random.randn(64)  # Simulated processed features
                cache.cache_neural_features(f"subject_{i:03d}", "P300", features)
            
            # Verify cache has data
            summary = cache.get_neural_cache_summary()
            assert summary["feature_entries"] >= 3
            
            # Test cache retrieval
            cached_features = cache.get_neural_features("subject_001", "P300")
            assert cached_features is not None
            
        finally:
            processor.stop()
    
    @pytest.mark.asyncio
    async def test_connection_pool_with_load_balancer_integration(self):
        """Test connection pool integration with load balancer."""
        # Create connection pool
        def create_test_connection():
            return {"id": time.time(), "active": True}
        
        pool = ConnectionPool(
            connection_factory=create_test_connection,
            min_connections=2,
            max_connections=5
        )
        
        # Create load balancer with workers that use the pool
        lb = LoadBalancer()
        
        async def worker_processor(data):
            async with pool.get_connection_context() as conn:
                # Simulate work with connection
                await asyncio.sleep(0.01)
                return f"Processed {data} with connection {conn['id']}"
        
        # Add workers
        for i in range(3):
            worker = Worker(
                id=f"worker_{i}",
                endpoint=f"endpoint_{i}",
                processor_func=lambda data, i=i: asyncio.run(worker_processor(f"{data}_w{i}"))
            )
            lb.add_worker(worker)
        
        # Process requests through load balancer
        results = []
        for i in range(10):
            try:
                result = await lb.process_request(f"request_{i}")
                results.append(result)
            except Exception as e:
                print(f"Request {i} failed: {e}")
        
        # Verify results
        assert len(results) >= 5  # Should have processed several requests
        
        # Check pool stats
        pool_stats = pool.get_stats()
        assert pool_stats["total_created"] >= 2  # Should have created connections
        
        # Check load balancer stats
        lb_stats = lb.get_stats()
        assert lb_stats["total_requests"] >= 5
    
    def test_full_performance_stack_integration(self, mock_neural_data):
        """Test full performance stack integration."""
        # Initialize performance components
        cache = NeuralDataCache()
        processor = NeuralBatchProcessor()
        
        # Create connection pool for external services
        def create_mock_service():
            return Mock(process=lambda x: f"Processed: {x}")
        
        pool = ConnectionPool(
            connection_factory=create_mock_service,
            min_connections=1,
            max_connections=3
        )
        
        processor.start()
        
        try:
            # Simulate full pipeline
            session_data = {}
            
            # 1. Process raw neural data through batch processor
            for i in range(5):
                raw_data = mock_neural_data(paradigm="P300")
                processor.add_neural_data(raw_data, "raw_neural", "P300", quality_score=0.8)
                
                # Cache raw data for later retrieval
                cache.cache_raw_neural_data(f"session_{i}", raw_data, 250, 
                                          [f"CH{j+1}" for j in range(8)])
            
            # Wait for batch processing
            time.sleep(0.8)
            
            # 2. Extract and cache features
            for i in range(5):
                # Simulate feature extraction
                features = np.random.randn(64)
                cache.cache_neural_features(f"subject_{i}", "P300", features, 
                                          {"session": f"session_{i}"})
            
            # 3. Cache trained models
            for i in range(2):
                model_data = {
                    "weights": np.random.randn(64, 4),
                    "bias": np.random.randn(4),
                    "accuracy": 0.85 + i * 0.05
                }
                cache.cache_decoder_model(f"model_{i}", model_data, "P300")
            
            # Verify integration
            processor_stats = processor.get_neural_stats()
            cache_summary = cache.get_neural_cache_summary()
            pool_stats = pool.get_stats()
            
            assert processor_stats["total_neural_samples"] >= 5
            assert cache_summary["total_entries"] >= 10
            assert cache_summary["raw_data_entries"] >= 5
            assert cache_summary["feature_entries"] >= 5
            assert cache_summary["model_entries"] >= 2
            
        finally:
            processor.stop()
            pool.close()


@pytest.mark.integration
class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_bci_pipeline_simulation(self, mock_neural_data):
        """Test complete BCI pipeline from data acquisition to response."""
        # Initialize all components
        bridge = BCIBridge(device="Simulation", paradigm="P300")
        cache = NeuralDataCache()
        processor = NeuralBatchProcessor()
        metrics = BCIMetricsCollector()
        monitor = HealthMonitor(check_interval=0.1)
        
        # Mock Claude adapter
        mock_claude = Mock()
        async def mock_execute(intention, context=None):
            return ClaudeResponse(
                content=f"Response to: {intention.command}",
                reasoning="Simulated processing",
                confidence=0.9,
                safety_flags=[],
                processing_time_ms=150.0,
                tokens_used=75
            )
        mock_claude.execute = AsyncMock(side_effect=mock_execute)
        
        # Setup health monitoring
        def bridge_health_check():
            from bci_agent_bridge.monitoring.health_monitor import HealthCheck, HealthStatus
            return HealthCheck(
                name="bridge_health",
                status=HealthStatus.HEALTHY,
                message="Bridge is operational",
                last_check=time.time(),
                duration_ms=5.0
            )
        
        monitor.register_health_check("bridge", bridge_health_check)
        
        # Start components
        processor.start()
        await monitor.start_monitoring()
        
        try:
            # Simulate complete workflow
            results = []
            
            for session_id in range(3):
                # 1. Generate neural data
                raw_data = mock_neural_data(channels=8, samples=250, paradigm="P300")
                
                # 2. Create neural data object
                neural_data = NeuralData(
                    data=raw_data,
                    timestamp=time.time(),
                    channels=[f"CH{i+1}" for i in range(8)],
                    sampling_rate=250,
                    metadata={"session_id": session_id}
                )
                
                # 3. Record metrics
                metrics.record_neural_sample(8, 250, 0.8)
                
                # 4. Process through bridge (decode intention)
                intention = bridge.decode_intention(neural_data)
                
                # 5. Record decoding metrics
                metrics.record_decoding_event("P300", intention.confidence, 45.0)
                
                # 6. Cache neural features
                features = np.mean(raw_data, axis=1)
                cache.cache_neural_features(f"subject_{session_id}", "P300", features)
                
                # 7. Add to batch processor
                processor.add_neural_data(raw_data, "raw_neural", "P300", quality_score=0.8)
                
                # 8. Process through Claude (if confidence is high enough)
                if intention.confidence > 0.7:
                    claude_response = await mock_claude.execute(intention)
                    
                    # 9. Record Claude metrics
                    metrics.record_claude_interaction(
                        "medical", 
                        claude_response.tokens_used,
                        claude_response.processing_time_ms
                    )
                    
                    results.append({
                        "session_id": session_id,
                        "command": intention.command,
                        "confidence": intention.confidence,
                        "response": claude_response.content,
                        "processing_time": claude_response.processing_time_ms
                    })
            
            # Wait for batch processing to complete
            await asyncio.sleep(1.0)
            
            # Verify end-to-end results
            assert len(results) >= 1  # Should have processed at least some high-confidence intentions
            
            # Check all components have recorded activity
            processor_stats = processor.get_neural_stats()
            cache_summary = cache.get_neural_cache_summary()
            bci_metrics = metrics.get_bci_performance_summary()
            health_summary = monitor.get_health_summary()
            
            # Verify pipeline worked
            assert processor_stats["total_neural_samples"] >= 3
            assert cache_summary["feature_entries"] >= 3
            assert bci_metrics["neural_processing"]["samples_processed"] >= 3
            assert bci_metrics["decoding_performance"]["attempts"] >= 3
            
            # Print results summary
            print(f"\n{'='*50}")
            print("END-TO-END INTEGRATION TEST RESULTS")
            print(f"{'='*50}")
            print(f"Sessions processed: {len(results)}")
            print(f"Neural samples: {processor_stats['total_neural_samples']}")
            print(f"Cache entries: {cache_summary['total_entries']}")
            print(f"Avg confidence: {bci_metrics['decoding_performance']['confidence']}")
            print(f"Health status: {health_summary['overall_status']}")
            print(f"{'='*50}\n")
            
        finally:
            processor.stop()
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio  
    async def test_error_handling_and_recovery_integration(self, mock_neural_data):
        """Test error handling and recovery across integrated components."""
        # Initialize components with error-prone configurations
        processor = NeuralBatchProcessor()
        cache = NeuralDataCache()
        
        # Create failing connection pool
        failure_count = 0
        def failing_connection_factory():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise ConnectionError("Connection failed")
            return {"id": failure_count, "working": True}
        
        pool = ConnectionPool(
            connection_factory=failing_connection_factory,
            min_connections=1,
            max_connections=3,
            connection_timeout=1.0
        )
        
        # Start components
        processor.start()
        
        try:
            # Test graceful degradation
            errors_encountered = []
            
            # 1. Try to get connection (should fail initially, then recover)
            try:
                conn = await pool.get_connection()
                pool.return_connection(conn)
                print("✅ Connection pool recovered successfully")
            except Exception as e:
                errors_encountered.append(f"Connection pool: {str(e)}")
                print(f"❌ Connection pool failed: {e}")
            
            # 2. Process neural data (should work)
            try:
                for i in range(3):
                    data = mock_neural_data()
                    processor.add_neural_data(data, "raw_neural", "P300", quality_score=0.7)
                
                await asyncio.sleep(0.5)
                stats = processor.get_neural_stats()
                assert stats["total_neural_samples"] >= 3
                print("✅ Batch processor working normally")
            except Exception as e:
                errors_encountered.append(f"Batch processor: {str(e)}")
                print(f"❌ Batch processor failed: {e}")
            
            # 3. Cache operations (should work)
            try:
                # Cache some data
                features = np.random.randn(64)
                cache.cache_neural_features("test_subject", "P300", features)
                
                # Retrieve data
                retrieved = cache.get_neural_features("test_subject", "P300")
                assert retrieved is not None
                print("✅ Neural cache working normally")
            except Exception as e:
                errors_encountered.append(f"Cache: {str(e)}")
                print(f"❌ Cache failed: {e}")
            
            # 4. Test cache cleanup under stress
            try:
                # Fill cache with temporary data
                for i in range(10):
                    temp_data = np.random.randn(32, 100)
                    cache.cache_raw_neural_data(f"temp_session_{i}", temp_data, 250, ["CH1"])
                
                # Force cleanup
                cleaned = cache.cleanup_expired_neural_data()
                print(f"✅ Cache cleanup removed {cleaned} expired entries")
            except Exception as e:
                errors_encountered.append(f"Cache cleanup: {str(e)}")
                print(f"❌ Cache cleanup failed: {e}")
            
            # Summary
            success_rate = (4 - len(errors_encountered)) / 4 * 100
            print(f"\n🔍 Error handling test completed")
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Errors encountered: {len(errors_encountered)}")
            
            # Should have recovered from initial failures
            assert success_rate >= 75, f"Too many failures: {errors_encountered}"
            
        finally:
            processor.stop()
            pool.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])