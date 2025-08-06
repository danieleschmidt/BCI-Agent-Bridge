"""
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
from unittest.mock import Mock, AsyncMock

# Make tests work with asyncio
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_neural_data():
    """Generate mock neural data for testing."""
    def _generate_data(channels=8, samples=250, paradigm="P300"):
        data = np.random.randn(channels, samples)
        
        # Add paradigm-specific patterns
        if paradigm == "P300":
            # Add P300 component to some trials
            if np.random.random() < 0.3:
                p300_latency = 300  # ms
                sample_idx = int(p300_latency * samples / 1000)
                if sample_idx < samples:
                    data[0, sample_idx] += 5  # P300 amplitude
        
        elif paradigm == "MotorImagery":
            # Add mu rhythm modulation
            t = np.linspace(0, samples/250, samples)
            mu_rhythm = np.sin(2 * np.pi * 10 * t)  # 10 Hz
            data[0] += mu_rhythm * (0.5 + 0.5 * np.random.random())
        
        elif paradigm == "SSVEP":
            # Add steady-state response
            freqs = [6.0, 7.5, 8.57, 10.0]
            target_freq = np.random.choice(freqs)
            t = np.linspace(0, samples/250, samples)
            ssvep_signal = 2 * np.sin(2 * np.pi * target_freq * t)
            data[0] += ssvep_signal
        
        return data
    
    return _generate_data


@pytest.fixture
def mock_bci_bridge():
    """Create mock BCI bridge for testing."""
    bridge = Mock()
    
    # Mock device info
    bridge.get_device_info.return_value = {
        'device': 'Simulation',
        'channels': 8,
        'sampling_rate': 250,
        'paradigm': 'P300',
        'connected': True,
        'streaming': False
    }
    
    # Mock device properties
    bridge.device = Mock()
    bridge.device.value = 'Simulation'
    bridge.channels = 8
    bridge.sampling_rate = 250
    bridge.paradigm = Mock()
    bridge.paradigm.value = 'P300'
    bridge.is_streaming = False
    bridge.data_buffer = []
    
    # Mock decoder
    bridge.decoder = Mock()
    bridge.decoder.get_decoder_info.return_value = {
        'type': 'P300Decoder',
        'channels': 8,
        'sampling_rate': 250,
        'calibrated': True,
        'confidence_threshold': 0.7,
        'last_confidence': 0.8
    }
    
    # Mock methods
    bridge.get_buffer.return_value = np.random.randn(8, 100)
    bridge.calibrate.return_value = None
    bridge.stop_streaming.return_value = None
    
    return bridge


@pytest.fixture
def mock_claude_adapter():
    """Create mock Claude adapter for testing."""
    adapter = Mock()
    
    # Mock properties
    adapter.model = "claude-3-sonnet-20240229"
    adapter.safety_mode = Mock()
    adapter.safety_mode.value = "medical"
    
    # Mock client
    adapter.client = Mock()
    adapter.client.api_key = "test_api_key"
    
    # Mock execute method
    async def mock_execute(intention, context=None):
        from bci_agent_bridge.adapters.claude_flow import ClaudeResponse
        return ClaudeResponse(
            content=f"Processed: {intention.command}",
            reasoning="Mock processing",
            confidence=0.8,
            safety_flags=[],
            processing_time_ms=100.0,
            tokens_used=50
        )
    
    adapter.execute = AsyncMock(side_effect=mock_execute)
    
    # Mock other methods
    adapter.process_text.return_value = Mock()
    adapter.set_mode.return_value = None
    adapter.suggest_break.return_value = Mock()
    adapter.increase_engagement.return_value = Mock()
    adapter.get_conversation_history.return_value = []
    adapter.clear_history.return_value = None
    
    return adapter


@pytest.fixture
def mock_metrics_collector():
    """Create mock metrics collector for testing."""
    collector = Mock()
    
    # Mock properties
    collector.metrics = {}
    collector.counters = {}
    collector.gauges = {}
    collector.histograms = {}
    
    # Mock methods
    collector.record_metric.return_value = None
    collector.increment_counter.return_value = None
    collector.set_gauge.return_value = None
    collector.record_histogram.return_value = None
    
    collector.get_metric_summary.return_value = Mock(
        count=10,
        min_value=1.0,
        max_value=10.0,
        avg_value=5.0,
        last_value=7.0
    )
    
    collector.get_all_summaries.return_value = {}
    collector.export_metrics.return_value = "{}"
    
    return collector


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_alert():
    """Create sample alert for testing."""
    from bci_agent_bridge.monitoring.alert_manager import Alert, AlertSeverity, AlertStatus
    import time
    
    return Alert(
        id="test_alert_001",
        name="test_alert",
        severity=AlertSeverity.WARNING,
        status=AlertStatus.ACTIVE,
        message="This is a test alert",
        details={"test_param": "test_value"},
        created_at=time.time()
    )


@pytest.fixture
def sample_health_check():
    """Create sample health check for testing."""
    from bci_agent_bridge.monitoring.health_monitor import HealthCheck, HealthStatus
    import time
    
    return HealthCheck(
        name="test_health_check",
        status=HealthStatus.HEALTHY,
        message="System is healthy",
        last_check=time.time(),
        duration_ms=10.0,
        details={"cpu_usage": 25.0, "memory_usage": 512.0}
    )


@pytest.fixture
def sample_worker():
    """Create sample worker for testing."""
    from bci_agent_bridge.performance.load_balancer import Worker, WorkerState
    
    def test_processor(data):
        return f"Processed: {data}"
    
    return Worker(
        id="test_worker_001",
        endpoint="http://test-worker:8080",
        weight=1.0,
        max_connections=100,
        state=WorkerState.HEALTHY,
        processor_func=test_processor,
        metadata={"region": "test", "version": "1.0.0"}
    )


@pytest.fixture(autouse=True)
def cleanup_async_tasks():
    """Clean up async tasks after each test."""
    yield
    
    # Cancel any remaining tasks
    try:
        loop = asyncio.get_running_loop()
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        if tasks:
            for task in tasks:
                task.cancel()
            # Wait briefly for cancellation
            try:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            except:
                pass
    except RuntimeError:
        # No event loop running
        pass


@pytest.fixture
def neural_data_samples():
    """Generate various neural data samples for testing."""
    samples = {}
    
    # P300 data
    samples['p300'] = np.random.randn(8, 250)
    samples['p300'][0, 75] += 8  # P300 peak at 300ms (75 samples at 250Hz)
    
    # Motor imagery data  
    samples['motor_imagery'] = np.random.randn(8, 500)
    t = np.linspace(0, 2, 500)  # 2 seconds
    mu_rhythm = np.sin(2 * np.pi * 10 * t)  # 10 Hz mu rhythm
    samples['motor_imagery'][2] += mu_rhythm * 2  # Add to C3 channel
    
    # SSVEP data
    samples['ssvep'] = np.random.randn(8, 1000)
    t = np.linspace(0, 4, 1000)  # 4 seconds
    ssvep_signal = np.sin(2 * np.pi * 7.5 * t)  # 7.5 Hz SSVEP
    samples['ssvep'][6] += ssvep_signal * 3  # Add to O1 channel
    
    # Noisy data
    samples['noisy'] = np.random.randn(8, 250) * 5  # High noise
    
    # Flat signal
    samples['flat'] = np.random.randn(8, 250)
    samples['flat'][0] = np.zeros(250)  # Completely flat channel
    
    # High amplitude artifacts
    samples['artifacts'] = np.random.randn(8, 250)
    samples['artifacts'][1, 50:55] = 200  # Large spike artifact
    
    return samples


@pytest.fixture(scope="session")
def test_config():
    """Test configuration parameters."""
    return {
        'test_timeout': 5.0,
        'neural_channels': 8,
        'sampling_rate': 250,
        'cache_size': 1024*1024,  # 1MB
        'batch_size': 10,
        'retry_attempts': 3,
        'circuit_breaker_threshold': 5
    }


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "neural: marks tests that require neural data processing"
    )
    config.addinivalue_line(
        "markers", "async_test: marks tests that use asyncio"
    )


# Async test helper
def run_async_test(coro):
    """Helper to run async tests in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_batch_items(count: int, data_type: str = "neural"):
        """Generate batch items for testing."""
        from bci_agent_bridge.performance.batch_processor import BatchItem
        import time
        
        items = []
        for i in range(count):
            if data_type == "neural":
                data = np.random.randn(8, 100)
            else:
                data = f"test_data_{i}"
            
            item = BatchItem(
                data=data,
                timestamp=time.time(),
                priority=i % 3,
                metadata={"index": i, "type": data_type}
            )
            items.append(item)
        
        return items
    
    @staticmethod
    def generate_cache_data(size_mb: float = 1.0):
        """Generate data for cache testing."""
        size_bytes = int(size_mb * 1024 * 1024)
        return np.random.bytes(size_bytes)
    
    @staticmethod
    def generate_metrics_data(count: int = 100):
        """Generate metrics data for testing."""
        from bci_agent_bridge.monitoring.metrics_collector import Metric
        import time
        
        metrics = []
        for i in range(count):
            metric = Metric(
                name=f"test_metric_{i % 10}",
                value=np.random.uniform(0, 100),
                timestamp=time.time() - i,
                tags={"index": str(i), "category": f"cat_{i % 3}"},
                unit="units"
            )
            metrics.append(metric)
        
        return metrics


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator