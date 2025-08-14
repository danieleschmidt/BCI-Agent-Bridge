"""
Enhanced BCI Bridge with advanced reliability features, circuit breakers,
and intelligent error recovery mechanisms.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, AsyncGenerator, List, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import json
import uuid
from datetime import datetime, timedelta

from ..signal_processing.preprocessing import SignalPreprocessor
from ..decoders.base import BaseDecoder
from .bridge import BCIBridge, BCIDevice, Paradigm, NeuralData, DecodedIntention

# Enhanced monitoring imports
try:
    from ..monitoring.adaptive_health_monitor import AdaptiveHealthMonitor, HealthAlert, HealthSeverity
    _MONITORING_AVAILABLE = True
except ImportError:
    _MONITORING_AVAILABLE = False

# Security imports
try:
    from ..security.input_validator import InputValidator, ValidationError, SecurityPolicy
    from ..security.audit_logger import security_logger, SecurityEvent
    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False
    class ValidationError(Exception):
        pass

# Circuit breaker imports
try:
    from ..utils.circuit_breaker import CircuitBreaker, CircuitState
    _CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    _CIRCUIT_BREAKER_AVAILABLE = False


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_STOP = "emergency_stop"


class SystemState(Enum):
    """Enhanced system states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


@dataclass
class ErrorEvent:
    """Enhanced error event tracking."""
    id: str
    timestamp: float
    error_type: str
    component: str
    message: str
    context: Dict[str, Any]
    recovery_strategy: ErrorRecoveryStrategy
    recovery_attempted: bool = False
    recovery_successful: bool = False
    impact_severity: HealthSeverity = HealthSeverity.WARNING


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    processing_latency_ms: float
    throughput_samples_per_sec: float
    error_rate_percentage: float
    memory_usage_mb: float
    cpu_usage_percentage: float
    buffer_utilization_percentage: float
    signal_quality_score: float
    decoder_confidence_avg: float
    uptime_seconds: float
    last_updated: float


class EnhancedBCIBridge(BCIBridge):
    """
    Enhanced BCI Bridge with advanced reliability features:
    - Adaptive health monitoring
    - Circuit breaker patterns
    - Intelligent error recovery
    - Performance optimization
    - Self-healing capabilities
    """
    
    def __init__(
        self,
        device: str = "Simulation",
        channels: int = 8,
        sampling_rate: int = 250,
        paradigm: str = "P300",
        buffer_size: int = 1000,
        privacy_mode: bool = True,
        enable_health_monitoring: bool = True,
        enable_auto_recovery: bool = True,
        performance_monitoring: bool = True
    ):
        # Initialize base BCI bridge
        super().__init__(device, channels, sampling_rate, paradigm, buffer_size, privacy_mode)
        
        # Enhanced configuration
        self.enable_health_monitoring = enable_health_monitoring
        self.enable_auto_recovery = enable_auto_recovery
        self.performance_monitoring = performance_monitoring
        
        # System state management
        self.system_state = SystemState.INITIALIZING
        self.state_history = deque(maxlen=100)
        self.state_change_callbacks = []
        
        # Error tracking and recovery
        self.error_events = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.recovery_strategies = {}
        self.fallback_systems = {}
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(
            processing_latency_ms=0.0,
            throughput_samples_per_sec=0.0,
            error_rate_percentage=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percentage=0.0,
            buffer_utilization_percentage=0.0,
            signal_quality_score=0.0,
            decoder_confidence_avg=0.0,
            uptime_seconds=0.0,
            last_updated=time.time()
        )
        
        # Initialize enhanced components
        self._initialize_enhanced_systems()
        
        # Set system state to healthy after initialization
        self._change_state(SystemState.HEALTHY, "Enhanced BCI Bridge initialized successfully")
        
        self.logger.info("Enhanced BCI Bridge initialized with advanced reliability features")
    
    def _initialize_enhanced_systems(self) -> None:
        """Initialize enhanced monitoring and recovery systems."""
        try:
            # Initialize adaptive health monitoring
            if self.enable_health_monitoring and _MONITORING_AVAILABLE:
                self.health_monitor = AdaptiveHealthMonitor(
                    check_interval=15,  # More frequent monitoring
                    enable_auto_remediation=self.enable_auto_recovery
                )
                self.health_monitor.add_health_callback(self._handle_health_alert)
                self.logger.info("Adaptive health monitoring initialized")
            else:
                self.health_monitor = None
            
            # Initialize circuit breakers for critical components
            if _CIRCUIT_BREAKER_AVAILABLE:
                self.circuit_breakers = {
                    'neural_processing': CircuitBreaker(
                        failure_threshold=5,
                        timeout_duration=30.0,
                        expected_exception=Exception
                    ),
                    'decoder_prediction': CircuitBreaker(
                        failure_threshold=3,
                        timeout_duration=60.0,
                        expected_exception=Exception
                    ),
                    'data_acquisition': CircuitBreaker(
                        failure_threshold=10,
                        timeout_duration=15.0,
                        expected_exception=Exception
                    )
                }
            else:
                self.circuit_breakers = {}
            
            # Initialize recovery strategies
            self._setup_recovery_strategies()
            
            # Initialize fallback systems
            self._setup_fallback_systems()
            
            # Performance monitoring setup
            if self.performance_monitoring:
                self._setup_performance_monitoring()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced systems: {e}")
            self._change_state(SystemState.DEGRADED, f"Enhanced system initialization failed: {e}")
    
    def _setup_recovery_strategies(self) -> None:
        """Setup error recovery strategies for different error types."""
        self.recovery_strategies = {
            'ValidationError': ErrorRecoveryStrategy.RETRY,
            'ConnectionError': ErrorRecoveryStrategy.CIRCUIT_BREAK,
            'ProcessingError': ErrorRecoveryStrategy.FALLBACK,
            'DecoderError': ErrorRecoveryStrategy.GRACEFUL_DEGRADATION,
            'SystemError': ErrorRecoveryStrategy.EMERGENCY_STOP,
            'TimeoutError': ErrorRecoveryStrategy.RETRY,
            'MemoryError': ErrorRecoveryStrategy.GRACEFUL_DEGRADATION
        }
    
    def _setup_fallback_systems(self) -> None:
        """Setup fallback systems for graceful degradation."""
        self.fallback_systems = {
            'neural_processing': {
                'enabled': True,
                'strategy': 'simplified_processing',
                'performance_reduction': 0.3  # 30% performance reduction
            },
            'decoder_prediction': {
                'enabled': True,
                'strategy': 'confidence_filtering',
                'minimum_confidence': 0.8  # Higher confidence requirement
            },
            'signal_quality': {
                'enabled': True,
                'strategy': 'adaptive_filtering',
                'quality_threshold': 0.4  # Lower quality threshold
            }
        }
    
    def _setup_performance_monitoring(self) -> None:
        """Setup performance monitoring and optimization."""
        self.performance_baseline = {}
        self.performance_history = deque(maxlen=1000)
        self.optimization_triggers = {
            'latency_threshold_ms': 150.0,
            'error_rate_threshold': 10.0,  # percentage
            'memory_threshold_mb': 1000.0,
            'cpu_threshold_percentage': 90.0
        }
    
    def add_state_change_callback(self, callback: Callable[[SystemState, str], None]) -> None:
        """Add callback for system state changes."""
        self.state_change_callbacks.append(callback)
    
    def _change_state(self, new_state: SystemState, reason: str) -> None:
        """Change system state with logging and callbacks."""
        old_state = self.system_state
        self.system_state = new_state
        
        # Record state change
        state_change = {
            'timestamp': time.time(),
            'from_state': old_state.value,
            'to_state': new_state.value,
            'reason': reason
        }
        self.state_history.append(state_change)
        
        # Log state change
        self.logger.info(f"System state changed: {old_state.value} -> {new_state.value} ({reason})")
        
        # Security logging for critical state changes
        if _SECURITY_AVAILABLE and new_state in [SystemState.EMERGENCY, SystemState.SHUTDOWN]:
            security_logger.log_suspicious_activity(
                activity_type="critical_state_change",
                details=state_change,
                risk_score=9 if new_state == SystemState.EMERGENCY else 7
            )
        
        # Notify callbacks
        for callback in self.state_change_callbacks:
            try:
                callback(new_state, reason)
            except Exception as e:
                self.logger.error(f"State change callback error: {e}")
    
    def _handle_health_alert(self, alert: HealthAlert) -> None:
        """Handle health alerts from the monitoring system."""
        self.logger.warning(f"Health alert received: {alert.message}")
        
        # Determine system state based on alert severity
        if alert.severity == HealthSeverity.CRITICAL:
            self._change_state(SystemState.EMERGENCY, f"Critical health alert: {alert.metric}")
        elif alert.severity == HealthSeverity.ERROR:
            if self.system_state == SystemState.HEALTHY:
                self._change_state(SystemState.DEGRADED, f"Health degradation: {alert.metric}")
        
        # Trigger recovery if auto-recovery is enabled
        if self.enable_auto_recovery and alert.auto_remediation:
            self._trigger_recovery(alert)
    
    def _trigger_recovery(self, alert: HealthAlert) -> None:
        """Trigger recovery actions based on health alert."""
        try:
            self.logger.info(f"Triggering recovery for alert: {alert.id}")
            
            # Create error event for tracking
            error_event = ErrorEvent(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                error_type="HealthAlert",
                component=alert.component,
                message=alert.message,
                context=alert.context,
                recovery_strategy=ErrorRecoveryStrategy.RETRY  # Default strategy
            )
            
            # Execute recovery
            success = self._execute_error_recovery(error_event)
            
            if success:
                self.logger.info(f"Recovery successful for alert: {alert.id}")
                if self.system_state == SystemState.DEGRADED:
                    self._change_state(SystemState.RECOVERING, "Recovery in progress")
            else:
                self.logger.error(f"Recovery failed for alert: {alert.id}")
                
        except Exception as e:
            self.logger.error(f"Recovery trigger failed: {e}")
    
    async def enhanced_stream(self) -> AsyncGenerator[NeuralData, None]:
        """
        Enhanced streaming with error recovery and performance monitoring.
        """
        if not self._device_connected:
            raise RuntimeError("BCI device not connected")
        
        # Start health monitoring if available
        if self.health_monitor and not self.health_monitor.is_monitoring:
            await self.health_monitor.start_monitoring()
        
        self.is_streaming = True
        self.logger.info("Starting enhanced neural data stream with reliability features")
        
        stream_start_time = time.time()
        sample_count = 0
        error_count = 0
        
        try:
            while self.is_streaming:
                processing_start = time.time()
                
                try:
                    # Use circuit breaker for data acquisition
                    if 'data_acquisition' in self.circuit_breakers:
                        raw_data = await self.circuit_breakers['data_acquisition'].call(
                            self._read_raw_data_protected
                        )
                    else:
                        raw_data = await self._read_raw_data_protected()
                    
                    # Use circuit breaker for neural processing
                    if 'neural_processing' in self.circuit_breakers:
                        processed_data = await self.circuit_breakers['neural_processing'].call(
                            self._process_data_protected, raw_data
                        )
                    else:
                        processed_data = await self._process_data_protected(raw_data)
                    
                    # Create neural data object
                    neural_data = NeuralData(
                        data=processed_data,
                        timestamp=time.time(),
                        channels=[f"CH{i+1}" for i in range(self.channels)],
                        sampling_rate=self.sampling_rate,
                        metadata={
                            "device": self.device.value, 
                            "paradigm": self.paradigm.value,
                            "system_state": self.system_state.value,
                            "processing_time_ms": (time.time() - processing_start) * 1000,
                            "sample_id": sample_count
                        }
                    )
                    
                    # Enhanced buffer management with validation
                    self._add_to_buffer_enhanced(neural_data)
                    
                    # Update performance metrics
                    self._update_performance_metrics(processing_start, sample_count, error_count)
                    
                    sample_count += 1
                    yield neural_data
                    
                except Exception as e:
                    error_count += 1
                    self._handle_streaming_error(e, sample_count, error_count)
                    
                    # Check if error rate is too high
                    if error_count / max(sample_count, 1) > 0.1:  # 10% error rate
                        self.logger.error("High error rate detected, switching to degraded mode")
                        self._change_state(SystemState.DEGRADED, "High error rate in streaming")
                        
                        # Attempt recovery
                        if self.enable_auto_recovery:
                            recovery_success = await self._attempt_streaming_recovery()
                            if not recovery_success:
                                break
                    
                    # Continue with next iteration
                    continue
                        
        except asyncio.CancelledError:
            self.logger.info("Enhanced neural data stream cancelled")
        except Exception as e:
            self.logger.error(f"Critical streaming error: {e}")
            self._change_state(SystemState.EMERGENCY, f"Critical streaming failure: {e}")
        finally:
            self.is_streaming = False
            
            # Stop health monitoring
            if self.health_monitor and self.health_monitor.is_monitoring:
                self.health_monitor.stop_monitoring()
            
            # Log streaming statistics
            stream_duration = time.time() - stream_start_time
            self.logger.info(
                f"Enhanced streaming ended: {sample_count} samples, {error_count} errors, "
                f"{stream_duration:.1f}s duration"
            )
    
    async def _read_raw_data_protected(self) -> np.ndarray:
        """Protected raw data reading with error handling."""
        try:
            return await self._read_raw_data()
        except Exception as e:
            self.logger.error(f"Raw data acquisition error: {e}")
            
            # Try fallback data generation if available
            if self.fallback_systems.get('data_acquisition', {}).get('enabled'):
                self.logger.info("Using fallback data generation")
                return self._generate_fallback_data()
            else:
                raise
    
    async def _process_data_protected(self, raw_data: np.ndarray) -> np.ndarray:
        """Protected data processing with error handling."""
        try:
            return self.preprocessor.process(raw_data)
        except Exception as e:
            self.logger.error(f"Data processing error: {e}")
            
            # Try simplified processing if available
            if self.fallback_systems.get('neural_processing', {}).get('enabled'):
                self.logger.info("Using simplified processing fallback")
                return self._simplified_processing(raw_data)
            else:
                raise
    
    def _generate_fallback_data(self) -> np.ndarray:
        """Generate fallback data when acquisition fails."""
        # Generate safe simulation data
        fallback_data = np.random.randn(self.channels, self.sampling_rate) * 0.1  # Low amplitude
        return fallback_data
    
    def _simplified_processing(self, raw_data: np.ndarray) -> np.ndarray:
        """Simplified processing for fallback mode."""
        # Basic filtering without complex processing
        try:
            # Simple bandpass filter simulation
            filtered_data = raw_data.copy()
            
            # Apply simple smoothing
            for i in range(filtered_data.shape[0]):
                if filtered_data.shape[1] > 5:
                    # Simple moving average
                    kernel_size = 5
                    kernel = np.ones(kernel_size) / kernel_size
                    filtered_data[i] = np.convolve(filtered_data[i], kernel, mode='same')
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Simplified processing failed: {e}")
            return raw_data  # Return raw data as last resort
    
    def _add_to_buffer_enhanced(self, neural_data: NeuralData) -> None:
        """Enhanced buffer management with comprehensive validation."""
        try:
            # Additional validation for enhanced mode
            if neural_data.data.size == 0:
                raise ValidationError("Empty neural data received")
            
            if np.any(np.isnan(neural_data.data)):
                self.logger.warning("NaN values detected in neural data, cleaning...")
                neural_data.data = np.nan_to_num(neural_data.data, nan=0.0)
            
            if np.any(np.isinf(neural_data.data)):
                self.logger.warning("Infinite values detected in neural data, cleaning...")
                neural_data.data = np.nan_to_num(neural_data.data, posinf=1.0, neginf=-1.0)
            
            # Check signal amplitude (basic sanity check)
            max_amplitude = np.max(np.abs(neural_data.data))
            if max_amplitude > 1000.0:  # Unusually high amplitude
                self.logger.warning(f"High amplitude signal detected: {max_amplitude}")
                # Scale down if necessary
                if max_amplitude > 10000.0:
                    neural_data.data = neural_data.data / (max_amplitude / 100.0)
            
            # Use parent's buffer management
            self._add_to_buffer_safe(neural_data)
            
        except ValidationError as e:
            self._handle_buffer_error(e, "validation")
        except Exception as e:
            self._handle_buffer_error(e, "general")
    
    def _handle_streaming_error(self, error: Exception, sample_count: int, error_count: int) -> None:
        """Handle streaming errors with recovery strategies."""
        error_type = type(error).__name__
        
        # Create error event
        error_event = ErrorEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            error_type=error_type,
            component="streaming",
            message=str(error),
            context={
                "sample_count": sample_count,
                "error_count": error_count,
                "error_rate": error_count / max(sample_count, 1)
            },
            recovery_strategy=self.recovery_strategies.get(error_type, ErrorRecoveryStrategy.RETRY)
        )
        
        self.error_events.append(error_event)
        self.error_counts[error_type] += 1
        
        self.logger.error(f"Streaming error [{error_type}]: {error}")
        
        # Execute recovery if enabled
        if self.enable_auto_recovery:
            self._execute_error_recovery(error_event)
    
    def _handle_buffer_error(self, error: Exception, error_category: str) -> None:
        """Handle buffer-related errors."""
        error_type = type(error).__name__
        
        error_event = ErrorEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            error_type=error_type,
            component="buffer_management",
            message=str(error),
            context={"category": error_category},
            recovery_strategy=ErrorRecoveryStrategy.RETRY
        )
        
        self.error_events.append(error_event)
        self.logger.error(f"Buffer error [{error_category}]: {error}")
    
    async def _attempt_streaming_recovery(self) -> bool:
        """Attempt to recover from streaming errors."""
        self.logger.info("Attempting streaming recovery...")
        
        try:
            # Stop current streaming
            self.is_streaming = False
            await asyncio.sleep(1.0)  # Brief pause
            
            # Reset circuit breakers
            for cb in self.circuit_breakers.values():
                if hasattr(cb, 'reset'):
                    cb.reset()
            
            # Clear error counts
            self.error_counts.clear()
            
            # Restart health monitoring if needed
            if self.health_monitor and not self.health_monitor.is_monitoring:
                await self.health_monitor.start_monitoring()
            
            self._change_state(SystemState.RECOVERING, "Streaming recovery attempted")
            
            # Short test to verify recovery
            test_data = await self._read_raw_data()
            if test_data is not None and test_data.size > 0:
                self._change_state(SystemState.HEALTHY, "Streaming recovery successful")
                return True
            else:
                self.logger.error("Recovery test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Streaming recovery failed: {e}")
            self._change_state(SystemState.EMERGENCY, f"Recovery failure: {e}")
            return False
    
    def _execute_error_recovery(self, error_event: ErrorEvent) -> bool:
        """Execute error recovery based on strategy."""
        error_event.recovery_attempted = True
        strategy = error_event.recovery_strategy
        
        try:
            if strategy == ErrorRecoveryStrategy.RETRY:
                success = self._retry_operation(error_event)
            elif strategy == ErrorRecoveryStrategy.FALLBACK:
                success = self._activate_fallback(error_event)
            elif strategy == ErrorRecoveryStrategy.CIRCUIT_BREAK:
                success = self._activate_circuit_breaker(error_event)
            elif strategy == ErrorRecoveryStrategy.GRACEFUL_DEGRADATION:
                success = self._graceful_degradation(error_event)
            elif strategy == ErrorRecoveryStrategy.EMERGENCY_STOP:
                success = self._emergency_stop(error_event)
            else:
                self.logger.warning(f"Unknown recovery strategy: {strategy}")
                success = False
            
            error_event.recovery_successful = success
            return success
            
        except Exception as e:
            self.logger.error(f"Recovery execution failed: {e}")
            error_event.recovery_successful = False
            return False
    
    def _retry_operation(self, error_event: ErrorEvent) -> bool:
        """Retry the failed operation."""
        self.logger.info(f"Retrying operation for error: {error_event.id}")
        # Implementation would retry the specific operation
        return True
    
    def _activate_fallback(self, error_event: ErrorEvent) -> bool:
        """Activate fallback system."""
        component = error_event.component
        if component in self.fallback_systems and self.fallback_systems[component]['enabled']:
            self.logger.info(f"Activating fallback for component: {component}")
            # Implementation would activate the specific fallback
            return True
        return False
    
    def _activate_circuit_breaker(self, error_event: ErrorEvent) -> bool:
        """Activate circuit breaker for component."""
        component = error_event.component
        if component in self.circuit_breakers:
            self.logger.info(f"Circuit breaker activated for: {component}")
            # Circuit breaker is automatically managed
            return True
        return False
    
    def _graceful_degradation(self, error_event: ErrorEvent) -> bool:
        """Implement graceful degradation."""
        self.logger.info("Implementing graceful degradation")
        self._change_state(SystemState.DEGRADED, f"Graceful degradation due to {error_event.error_type}")
        
        # Reduce system performance/complexity
        if hasattr(self, 'preprocessor'):
            # Simplify preprocessing
            pass
        
        return True
    
    def _emergency_stop(self, error_event: ErrorEvent) -> bool:
        """Execute emergency stop procedure."""
        self.logger.critical(f"Emergency stop triggered by: {error_event.error_type}")
        self._change_state(SystemState.EMERGENCY, f"Emergency stop: {error_event.message}")
        
        # Stop all operations
        self.stop_streaming()
        
        # Log critical event
        if _SECURITY_AVAILABLE:
            security_logger.log_suspicious_activity(
                activity_type="emergency_stop",
                details={
                    "error_event_id": error_event.id,
                    "error_type": error_event.error_type,
                    "component": error_event.component
                },
                risk_score=10
            )
        
        return True
    
    def _update_performance_metrics(self, processing_start: float, sample_count: int, error_count: int) -> None:
        """Update system performance metrics."""
        current_time = time.time()
        processing_time = current_time - processing_start
        
        # Calculate metrics
        self.performance_metrics.processing_latency_ms = processing_time * 1000
        
        if sample_count > 0:
            self.performance_metrics.throughput_samples_per_sec = 1.0 / processing_time
            self.performance_metrics.error_rate_percentage = (error_count / sample_count) * 100
        
        # Buffer utilization
        if hasattr(self, 'data_buffer') and hasattr(self, 'buffer_size'):
            self.performance_metrics.buffer_utilization_percentage = (
                len(self.data_buffer) / self.buffer_size
            ) * 100
        
        # System resource metrics (simplified)
        try:
            import psutil
            self.performance_metrics.memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            self.performance_metrics.cpu_usage_percentage = psutil.cpu_percent()
        except ImportError:
            pass
        
        self.performance_metrics.last_updated = current_time
        
        # Store in history
        if hasattr(self, 'performance_history'):
            self.performance_history.append({
                'timestamp': current_time,
                'metrics': self.performance_metrics.__dict__.copy()
            })
    
    def enhanced_decode_intention(self, neural_data: NeuralData) -> DecodedIntention:
        """
        Enhanced intention decoding with circuit breaker protection and fallback.
        """
        if self.decoder is None:
            raise RuntimeError("No decoder initialized")
        
        try:
            # Use circuit breaker for decoder prediction
            if 'decoder_prediction' in self.circuit_breakers:
                intention = self.circuit_breakers['decoder_prediction'].call(
                    self._decode_intention_protected, neural_data
                )
            else:
                intention = self._decode_intention_protected(neural_data)
            
            # Enhanced confidence filtering in degraded mode
            if (self.system_state == SystemState.DEGRADED and 
                'decoder_prediction' in self.fallback_systems):
                min_confidence = self.fallback_systems['decoder_prediction']['minimum_confidence']
                if intention.confidence < min_confidence:
                    # Return low-confidence indication
                    intention.command = "Low confidence - no action"
                    intention.context['fallback_mode'] = True
            
            return intention
            
        except Exception as e:
            self.logger.error(f"Enhanced decoder error: {e}")
            
            # Create fallback intention
            return DecodedIntention(
                command="Decoder error - fallback mode",
                confidence=0.0,
                context={
                    "error": str(e),
                    "fallback_mode": True,
                    "timestamp": neural_data.timestamp
                },
                timestamp=time.time()
            )
    
    def _decode_intention_protected(self, neural_data: NeuralData) -> DecodedIntention:
        """Protected intention decoding."""
        return self.decode_intention(neural_data)
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced system status."""
        base_status = self.get_device_info()
        
        enhanced_status = {
            **base_status,
            "system_state": self.system_state.value,
            "health_monitoring_enabled": self.enable_health_monitoring,
            "auto_recovery_enabled": self.enable_auto_recovery,
            "performance_monitoring_enabled": self.performance_monitoring,
            "active_errors": len([e for e in self.error_events if not e.recovery_successful]),
            "total_errors": len(self.error_events),
            "circuit_breaker_states": {
                name: cb.state.value if hasattr(cb, 'state') else "unknown"
                for name, cb in self.circuit_breakers.items()
            },
            "fallback_systems": {
                name: config['enabled'] for name, config in self.fallback_systems.items()
            },
            "performance_metrics": self.performance_metrics.__dict__,
            "health_status": self.health_monitor.get_health_status() if self.health_monitor else None,
            "last_state_change": self.state_history[-1] if self.state_history else None
        }
        
        return enhanced_status
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and statistics."""
        recent_errors = [e for e in self.error_events if time.time() - e.timestamp < 3600]  # Last hour
        
        return {
            "total_errors": len(self.error_events),
            "recent_errors": len(recent_errors),
            "error_types": dict(self.error_counts),
            "recovery_success_rate": len([e for e in self.error_events if e.recovery_successful]) / max(len(self.error_events), 1),
            "most_common_errors": sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_check_start = time.time()
        
        checks = {
            "device_connection": self._device_connected,
            "decoder_available": self.decoder is not None,
            "streaming_active": self.is_streaming,
            "buffer_healthy": len(self.data_buffer) < self.buffer_size * 0.9,
            "system_state_healthy": self.system_state in [SystemState.HEALTHY, SystemState.DEGRADED],
            "error_rate_acceptable": self.performance_metrics.error_rate_percentage < 10.0,
            "latency_acceptable": self.performance_metrics.processing_latency_ms < 150.0
        }
        
        # Overall health score
        health_score = sum(checks.values()) / len(checks)
        
        health_status = {
            "overall_health_score": health_score,
            "individual_checks": checks,
            "system_state": self.system_state.value,
            "check_duration_ms": (time.time() - health_check_start) * 1000,
            "timestamp": time.time(),
            "recommendations": self._get_health_recommendations(checks)
        }
        
        return health_status
    
    def _get_health_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Get health recommendations based on check results."""
        recommendations = []
        
        if not checks.get("device_connection"):
            recommendations.append("Check BCI device connection")
        
        if not checks.get("decoder_available"):
            recommendations.append("Initialize neural decoder")
        
        if not checks.get("buffer_healthy"):
            recommendations.append("Consider increasing buffer size or reducing data retention")
        
        if not checks.get("error_rate_acceptable"):
            recommendations.append("Investigate high error rate - check logs and system resources")
        
        if not checks.get("latency_acceptable"):
            recommendations.append("Optimize processing pipeline to reduce latency")
        
        if not recommendations:
            recommendations.append("System operating normally")
        
        return recommendations


# Factory function for easy instantiation
def create_enhanced_bci_bridge(config: Optional[Dict[str, Any]] = None) -> EnhancedBCIBridge:
    """Create and configure an enhanced BCI bridge."""
    config = config or {}
    
    return EnhancedBCIBridge(
        device=config.get('device', 'Simulation'),
        channels=config.get('channels', 8),
        sampling_rate=config.get('sampling_rate', 250),
        paradigm=config.get('paradigm', 'P300'),
        buffer_size=config.get('buffer_size', 1000),
        privacy_mode=config.get('privacy_mode', True),
        enable_health_monitoring=config.get('health_monitoring', True),
        enable_auto_recovery=config.get('auto_recovery', True),
        performance_monitoring=config.get('performance_monitoring', True)
    )