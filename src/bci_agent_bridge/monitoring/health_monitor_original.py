"""
Health monitoring system for BCI bridge components.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    last_check: float
    duration_ms: float
    details: Dict[str, Any] = None


class HealthMonitor:
    """
    Monitors the health of BCI bridge components with automatic recovery.
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        self.health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.last_results: Dict[str, HealthCheck] = {}
        self.monitoring_active = False
        self.recovery_strategies: Dict[str, Callable] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def register_recovery_strategy(self, component_name: str, recovery_func: Callable) -> None:
        """Register a recovery strategy for a component."""
        self.recovery_strategies[component_name] = recovery_func
        self.logger.info(f"Registered recovery strategy for: {component_name}")
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.monitoring_active:
            self.logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self.run_all_checks()
                await self.handle_unhealthy_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = await asyncio.to_thread(check_func)
                result.duration_ms = (time.time() - start_time) * 1000
                result.last_check = time.time()
                results[name] = result
                self.last_results[name] = result
            except Exception as e:
                self.logger.error(f"Health check '{name}' failed: {e}")
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(e)}",
                    last_check=time.time(),
                    duration_ms=0.0,
                    details={"error": str(e)}
                )
        
        return results
    
    async def handle_unhealthy_components(self) -> None:
        """Attempt recovery for unhealthy components."""
        for name, result in self.last_results.items():
            if result.status == HealthStatus.UNHEALTHY:
                self.logger.warning(f"Component '{name}' is unhealthy: {result.message}")
                
                # Attempt recovery if strategy is available
                if name in self.recovery_strategies:
                    try:
                        self.logger.info(f"Attempting recovery for '{name}'")
                        await asyncio.to_thread(self.recovery_strategies[name])
                        self.logger.info(f"Recovery attempted for '{name}'")
                    except Exception as e:
                        self.logger.error(f"Recovery failed for '{name}': {e}")
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        elif any(s == HealthStatus.UNKNOWN for s in statuses):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        overall_status = self.get_overall_health()
        
        return {
            "overall_status": overall_status.value,
            "monitoring_active": self.monitoring_active,
            "check_interval": self.check_interval,
            "components": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "last_check": result.last_check,
                    "duration_ms": result.duration_ms,
                    "details": result.details or {}
                }
                for name, result in self.last_results.items()
            },
            "summary": {
                "total_components": len(self.last_results),
                "healthy": sum(1 for r in self.last_results.values() if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in self.last_results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNHEALTHY),
                "unknown": sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNKNOWN)
            }
        }


def create_bci_health_checks(bci_bridge) -> Dict[str, Callable]:
    """Create standard health checks for BCI components."""
    
    def check_bci_device() -> HealthCheck:
        """Check BCI device connectivity."""
        try:
            device_info = bci_bridge.get_device_info()
            if device_info.get('connected', False):
                return HealthCheck(
                    name="bci_device",
                    status=HealthStatus.HEALTHY,
                    message="BCI device connected and responsive",
                    last_check=time.time(),
                    duration_ms=0.0,
                    details=device_info
                )
            else:
                return HealthCheck(
                    name="bci_device",
                    status=HealthStatus.UNHEALTHY,
                    message="BCI device not connected",
                    last_check=time.time(),
                    duration_ms=0.0,
                    details=device_info
                )
        except Exception as e:
            return HealthCheck(
                name="bci_device",
                status=HealthStatus.UNKNOWN,
                message=f"Device check failed: {str(e)}",
                last_check=time.time(),
                duration_ms=0.0
            )
    
    def check_signal_quality() -> HealthCheck:
        """Check neural signal quality."""
        try:
            buffer_size = len(bci_bridge.data_buffer)
            if buffer_size > 0:
                # Check if we're receiving data
                recent_data = bci_bridge.get_buffer(min(10, buffer_size))
                
                if recent_data.size > 0:
                    # Simple quality metrics
                    noise_level = recent_data.std()
                    signal_range = recent_data.max() - recent_data.min()
                    
                    if noise_level < 100 and signal_range > 1:  # Reasonable values
                        return HealthCheck(
                            name="signal_quality",
                            status=HealthStatus.HEALTHY,
                            message="Signal quality is good",
                            last_check=time.time(),
                            duration_ms=0.0,
                            details={
                                "noise_level": float(noise_level),
                                "signal_range": float(signal_range),
                                "buffer_size": buffer_size
                            }
                        )
                    else:
                        return HealthCheck(
                            name="signal_quality",
                            status=HealthStatus.DEGRADED,
                            message="Signal quality issues detected",
                            last_check=time.time(),
                            duration_ms=0.0,
                            details={
                                "noise_level": float(noise_level),
                                "signal_range": float(signal_range),
                                "buffer_size": buffer_size
                            }
                        )
                else:
                    return HealthCheck(
                        name="signal_quality",
                        status=HealthStatus.UNHEALTHY,
                        message="No signal data available",
                        last_check=time.time(),
                        duration_ms=0.0
                    )
            else:
                return HealthCheck(
                    name="signal_quality",
                    status=HealthStatus.DEGRADED,
                    message="Buffer is empty",
                    last_check=time.time(),
                    duration_ms=0.0
                )
        except Exception as e:
            return HealthCheck(
                name="signal_quality",
                status=HealthStatus.UNKNOWN,
                message=f"Signal quality check failed: {str(e)}",
                last_check=time.time(),
                duration_ms=0.0
            )
    
    def check_decoder_status() -> HealthCheck:
        """Check neural decoder status."""
        try:
            if bci_bridge.decoder is None:
                return HealthCheck(
                    name="decoder_status",
                    status=HealthStatus.UNHEALTHY,
                    message="No decoder initialized",
                    last_check=time.time(),
                    duration_ms=0.0
                )
            
            decoder_info = bci_bridge.decoder.get_decoder_info()
            
            if decoder_info.get('calibrated', False):
                confidence = decoder_info.get('last_confidence', 0.0)
                if confidence >= 0.7:
                    status = HealthStatus.HEALTHY
                    message = "Decoder operating with high confidence"
                elif confidence >= 0.5:
                    status = HealthStatus.DEGRADED
                    message = "Decoder operating with moderate confidence"
                else:
                    status = HealthStatus.DEGRADED
                    message = "Decoder confidence is low"
            else:
                status = HealthStatus.DEGRADED
                message = "Decoder not calibrated"
            
            return HealthCheck(
                name="decoder_status",
                status=status,
                message=message,
                last_check=time.time(),
                duration_ms=0.0,
                details=decoder_info
            )
            
        except Exception as e:
            return HealthCheck(
                name="decoder_status",
                status=HealthStatus.UNKNOWN,
                message=f"Decoder check failed: {str(e)}",
                last_check=time.time(),
                duration_ms=0.0
            )
    
    return {
        "bci_device": check_bci_device,
        "signal_quality": check_signal_quality,
        "decoder_status": check_decoder_status
    }


def create_claude_health_checks(claude_adapter) -> Dict[str, Callable]:
    """Create health checks for Claude adapter."""
    
    def check_claude_connectivity() -> HealthCheck:
        """Check Claude API connectivity."""
        try:
            # Simple connectivity test
            test_intention = type('TestIntention', (), {
                'command': 'health check',
                'confidence': 1.0,
                'context': {'test': True},
                'timestamp': time.time()
            })()
            
            # This would need to be modified to not actually call the API
            # For now, we'll just check if the adapter is properly configured
            if hasattr(claude_adapter, 'client') and claude_adapter.client:
                return HealthCheck(
                    name="claude_connectivity",
                    status=HealthStatus.HEALTHY,
                    message="Claude adapter is configured and ready",
                    last_check=time.time(),
                    duration_ms=0.0,
                    details={
                        "model": claude_adapter.model,
                        "safety_mode": claude_adapter.safety_mode.value
                    }
                )
            else:
                return HealthCheck(
                    name="claude_connectivity",
                    status=HealthStatus.UNHEALTHY,
                    message="Claude adapter not properly configured",
                    last_check=time.time(),
                    duration_ms=0.0
                )
        except Exception as e:
            return HealthCheck(
                name="claude_connectivity",
                status=HealthStatus.UNKNOWN,
                message=f"Claude connectivity check failed: {str(e)}",
                last_check=time.time(),
                duration_ms=0.0
            )
    
    return {
        "claude_connectivity": check_claude_connectivity
    }