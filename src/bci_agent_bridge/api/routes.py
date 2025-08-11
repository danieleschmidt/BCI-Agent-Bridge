"""
Enhanced API routes for BCI-Agent-Bridge with comprehensive security, monitoring, and error handling.
Production-ready medical-grade API with full compliance, audit logging, and performance monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, WebSocket, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator, constr, confloat
from typing import Dict, Any, List, Optional, Union
from functools import wraps
import time
import asyncio
import logging
import uuid
import json
from datetime import datetime, timedelta
from collections import defaultdict

from ..core.bridge import BCIBridge, NeuralData, DecodedIntention
from ..adapters.claude_flow import ClaudeFlowAdapter, ClaudeResponse

# Security imports
try:
    from ..security.input_validator import InputValidator, ValidationError, SecurityPolicy
    from ..security.audit_logger import security_logger, SecurityEvent
    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False

router = APIRouter()
logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Rate limiting storage
rate_limit_storage = defaultdict(list)

# Global metrics collection
metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "avg_response_time_ms": 0.0,
    "neural_processing_total": 0,
    "claude_executions_total": 0
}

class BaseAPIModel(BaseModel):
    """Base model with common configuration for all API models."""
    
    class Config:
        validate_all = True
        use_enum_values = True
        str_strip_whitespace = True
        extra = "forbid"  # Prevent extra fields for security


class ValidationErrorResponse(BaseModel):
    """Standard validation error response."""
    detail: str
    error_type: str = "validation_error"
    request_id: str
    timestamp: float
    field_errors: Optional[List[Dict[str, str]]] = None


class ErrorResponse(BaseAPIModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Standardized error code")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: float = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Enhanced Pydantic models with comprehensive validation
class HealthResponse(BaseAPIModel):
    status: constr(regex="^(healthy|degraded|critical)$") = Field(..., description="Overall system health")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(default="0.2.0", description="API version")
    components: Dict[str, str] = Field(..., description="Component-level health status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    request_id: str = Field(..., description="Unique request identifier")


class NeuralDataRequest(BaseAPIModel):
    channels: int = Field(..., ge=1, le=256, description="Number of EEG channels (1-256)")
    sampling_rate: int = Field(..., ge=1, le=8000, description="Sampling rate in Hz (1-8000)")
    paradigm: constr(regex="^(P300|MotorImagery|SSVEP|Hybrid)$") = Field(..., description="BCI paradigm")
    duration_seconds: confloat(gt=0.0, le=60.0) = Field(default=1.0, description="Data window duration")
    subject_id: Optional[str] = Field(None, regex="^[a-zA-Z0-9_-]{1,50}$", description="Subject identifier")
    session_id: Optional[str] = Field(None, regex="^[a-zA-Z0-9_-]{1,50}$", description="Session identifier")
    quality_threshold: confloat(ge=0.0, le=1.0) = Field(default=0.5, description="Signal quality threshold")

    @validator('paradigm')
    def validate_paradigm(cls, v):
        valid_paradigms = ['P300', 'MotorImagery', 'SSVEP', 'Hybrid']
        if v not in valid_paradigms:
            raise ValueError(f'Paradigm must be one of: {valid_paradigms}')
        return v


class IntentionRequest(BaseAPIModel):
    command: constr(min_length=1, max_length=500) = Field(..., description="Neural command text")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence level (0.0-1.0)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    priority: constr(regex="^(low|normal|high|critical)$") = Field(default="normal", description="Processing priority")
    timeout_seconds: confloat(gt=0.0, le=30.0) = Field(default=10.0, description="Processing timeout")

    @validator('command')
    def sanitize_command(cls, v):
        # Basic XSS/injection protection
        prohibited_patterns = ['<script', 'javascript:', 'data:', 'vbscript:']
        v_lower = v.lower()
        for pattern in prohibited_patterns:
            if pattern in v_lower:
                raise ValueError(f'Command contains prohibited pattern: {pattern}')
        return v


class CalibrationRequest(BaseAPIModel):
    paradigm: constr(regex="^(P300|MotorImagery|SSVEP|Hybrid)$") = Field(..., description="Calibration paradigm")
    trials: int = Field(..., ge=10, le=200, description="Number of calibration trials")
    subject_id: constr(regex="^[a-zA-Z0-9_-]{1,50}$") = Field(..., description="Subject identifier")
    session_notes: Optional[str] = Field(None, max_length=1000, description="Session notes")
    baseline_duration: confloat(gt=0.0, le=60.0) = Field(default=30.0, description="Baseline recording duration")


class StreamingResponse(BaseAPIModel):
    data: List[List[float]] = Field(..., description="Neural signal data matrix")
    timestamp: float = Field(..., description="Data timestamp")
    channels: List[str] = Field(..., description="Channel names")
    confidence: Optional[float] = Field(None, description="Signal quality confidence")
    signal_quality: Dict[str, float] = Field(default_factory=dict, description="Per-channel quality metrics")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class MetricsResponse(BaseAPIModel):
    system_metrics: Dict[str, Union[int, float, str]] = Field(..., description="System-level metrics")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    neural_metrics: Dict[str, Union[int, float]] = Field(..., description="Neural processing metrics")
    timestamp: float = Field(..., description="Metrics timestamp")
    collection_duration_ms: float = Field(..., description="Metrics collection time")


# Rate limiting decorator
def rate_limit(max_requests: int = 100, window_minutes: int = 1):
    """Rate limiting decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host if request.client else "unknown"
            current_time = time.time()
            window_start = current_time - (window_minutes * 60)
            
            # Clean old entries
            rate_limit_storage[client_ip] = [
                req_time for req_time in rate_limit_storage[client_ip] 
                if req_time > window_start
            ]
            
            # Check rate limit
            if len(rate_limit_storage[client_ip]) >= max_requests:
                if _SECURITY_AVAILABLE:
                    security_logger.log_suspicious_activity(
                        activity_type="rate_limit_exceeded",
                        details={"client_ip": client_ip, "requests": len(rate_limit_storage[client_ip])},
                        risk_score=6
                    )
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            rate_limit_storage[client_ip].append(current_time)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


# Error handling decorator
def handle_errors(timeout_seconds: float = 30.0):
    """Global error handling decorator with timeout protection."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            try:
                # Add timeout protection
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                
                # Update success metrics
                metrics["requests_success"] += 1
                processing_time = (time.time() - start_time) * 1000
                metrics["avg_response_time_ms"] = (
                    metrics["avg_response_time_ms"] * 0.9 + processing_time * 0.1
                )
                
                # Add request ID to response if it's a dict
                if isinstance(result, dict):
                    result["request_id"] = request_id
                    result["processing_time_ms"] = processing_time
                
                return result
                
            except asyncio.TimeoutError:
                metrics["requests_error"] += 1
                logger.error(f"Request {request_id} timed out after {timeout_seconds}s")
                raise HTTPException(status_code=408, detail=f"Request timed out after {timeout_seconds} seconds")
            
            except HTTPException:
                metrics["requests_error"] += 1
                raise
            
            except Exception as e:
                metrics["requests_error"] += 1
                logger.error(f"Request {request_id} failed: {e}")
                if _SECURITY_AVAILABLE:
                    security_logger.log_system_error(
                        component="api_routes",
                        error_type=type(e).__name__,
                        error_message=str(e)
                    )
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
            
            finally:
                metrics["requests_total"] += 1
                
        return wrapper
    return decorator


# Security validation
async def validate_request_size(request: Request):
    """Validate request size to prevent DoS attacks."""
    if hasattr(request, 'body'):
        try:
            body = await request.body()
            if len(body) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=413, detail="Request too large")
        except Exception:
            pass


# Enhanced dependency injection with security validation
async def get_bci_bridge(request: Request) -> BCIBridge:
    """Get BCI bridge from application state with security validation."""
    await validate_request_size(request)
    
    if not hasattr(request.app.state, 'bci_bridge') or request.app.state.bci_bridge is None:
        logger.error("BCI Bridge not initialized in application state")
        raise HTTPException(status_code=503, detail="BCI Bridge service unavailable")
    
    bridge = request.app.state.bci_bridge
    
    # Validate bridge health
    try:
        health_status = bridge.get_health_status()
        if health_status.get("status") == "error":
            raise HTTPException(status_code=503, detail="BCI Bridge in error state")
    except Exception as e:
        logger.error(f"BCI Bridge health check failed: {e}")
        raise HTTPException(status_code=503, detail="BCI Bridge health check failed")
    
    return bridge


async def get_claude_adapter(request: Request) -> ClaudeFlowAdapter:
    """Get Claude adapter from application state with security validation."""
    if not hasattr(request.app.state, 'claude_adapter') or request.app.state.claude_adapter is None:
        logger.error("Claude adapter not initialized in application state")
        raise HTTPException(status_code=503, detail="Claude AI service unavailable")
    
    return request.app.state.claude_adapter


async def get_input_validator() -> InputValidator:
    """Get input validator with clinical security policy."""
    if _SECURITY_AVAILABLE:
        return InputValidator(SecurityPolicy.CLINICAL)
    else:
        # Fallback validator if security module not available
        class FallbackValidator:
            def validate_neural_data(self, *args, **kwargs): pass
            def validate_string_input(self, text, field_name="input"): return text
        return FallbackValidator()


# Add security headers middleware
async def add_security_headers(request: Request, response: Response):
    """Add comprehensive security headers."""
    security_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; object-src 'none';",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
    }
    
    for header, value in security_headers.items():
        response.headers[header] = value


# Enhanced health and status endpoints
@router.get("/health", response_model=HealthResponse)
@handle_errors(timeout_seconds=10.0)
@rate_limit(max_requests=200, window_minutes=1)
async def enhanced_health_check(
    request: Request,
    bci_bridge: BCIBridge = Depends(get_bci_bridge),
    claude_adapter: ClaudeFlowAdapter = Depends(get_claude_adapter)
):
    """Comprehensive health check endpoint with component-level monitoring."""
    request_id = str(uuid.uuid4())
    timestamp = time.time()
    
    try:
        # Check BCI bridge health
        device_info = bci_bridge.get_device_info()
        bridge_health = bci_bridge.get_health_status()
        
        # Component health assessment
        components = {
            "bci_bridge": "healthy" if device_info.get("connected") and bridge_health.get("status") != "error" else "degraded",
            "claude_adapter": "healthy" if claude_adapter else "critical",
            "neural_decoder": "healthy" if bci_bridge.decoder else "degraded",
            "security_system": "healthy" if _SECURITY_AVAILABLE else "degraded",
            "data_pipeline": "healthy" if len(bci_bridge.data_buffer) < bci_bridge.buffer_size * 0.9 else "degraded"
        }
        
        # Determine overall status
        if "critical" in components.values():
            overall_status = "critical"
        elif "degraded" in components.values():
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Log health check
        if _SECURITY_AVAILABLE:
            security_logger.log_security_event(
                event_type=SecurityEvent.HEALTH_CHECK,
                resource="api_health",
                action="check",
                details={
                    "status": overall_status,
                    "components": components,
                    "request_id": request_id
                },
                risk_score=1
            )
        
        return HealthResponse(
            status=overall_status,
            timestamp=timestamp,
            components=components,
            uptime_seconds=timestamp - bridge_health.get("timestamp", timestamp),
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@router.get("/ready")
@handle_errors(timeout_seconds=5.0)
async def readiness_probe(
    request: Request,
    bci_bridge: BCIBridge = Depends(get_bci_bridge)
):
    """Kubernetes readiness probe endpoint."""
    try:
        # Quick readiness checks
        device_info = bci_bridge.get_device_info()
        
        if not device_info.get("connected"):
            raise HTTPException(status_code=503, detail="BCI device not ready")
        
        return {"status": "ready", "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


# Enhanced BCI data endpoints with security and monitoring
@router.post("/bci/start")
@handle_errors(timeout_seconds=15.0)
@rate_limit(max_requests=10, window_minutes=1)
async def start_enhanced_bci_streaming(
    request: Request,
    background_tasks: BackgroundTasks,
    bci_bridge: BCIBridge = Depends(get_bci_bridge),
    validator: InputValidator = Depends(get_input_validator)
):
    """Enhanced BCI data streaming with monitoring and error recovery."""
    request_id = str(uuid.uuid4())
    
    try:
        if bci_bridge.is_streaming:
            return {
                "message": "BCI streaming already active",
                "status": "active",
                "request_id": request_id,
                "timestamp": time.time()
            }
        
        # Enhanced streaming handler with error recovery
        async def enhanced_stream_handler():
            """Enhanced streaming with monitoring and error recovery."""
            error_count = 0
            max_errors = 10
            last_error_time = 0
            
            try:
                async for neural_data in bci_bridge.stream():
                    try:
                        # Validate neural data
                        validator.validate_neural_data(
                            neural_data.data, 
                            bci_bridge.channels, 
                            bci_bridge.sampling_rate
                        )
                        
                        # Update metrics
                        metrics["neural_processing_total"] += 1
                        
                        # Log successful processing
                        if time.time() - last_error_time > 60:  # Reset error count after 1 minute
                            error_count = 0
                        
                        logger.debug(f"Neural data processed: {neural_data.timestamp}")
                        
                    except ValidationError as e:
                        error_count += 1
                        last_error_time = time.time()
                        logger.warning(f"Neural data validation failed: {e}")
                        
                        if error_count >= max_errors:
                            logger.error("Too many validation errors, stopping stream")
                            bci_bridge.stop_streaming()
                            break
                    
                    except Exception as e:
                        error_count += 1
                        last_error_time = time.time()
                        logger.error(f"Neural processing error: {e}")
                        
                        if error_count >= max_errors:
                            logger.error("Too many processing errors, stopping stream")
                            bci_bridge.stop_streaming()
                            break
            
            except Exception as e:
                logger.error(f"Streaming handler failed: {e}")
                if _SECURITY_AVAILABLE:
                    security_logger.log_system_error(
                        component="bci_streaming",
                        error_type="stream_failure",
                        error_message=str(e)
                    )
        
        background_tasks.add_task(enhanced_stream_handler)
        
        # Log stream start
        if _SECURITY_AVAILABLE:
            security_logger.log_security_event(
                event_type=SecurityEvent.CONFIGURATION_CHANGE,
                resource="bci_streaming",
                action="start",
                details={"request_id": request_id},
                risk_score=3
            )
        
        return {
            "message": "Enhanced BCI streaming started with monitoring",
            "status": "active",
            "request_id": request_id,
            "timestamp": time.time(),
            "features": ["error_recovery", "validation", "monitoring"]
        }
        
    except Exception as e:
        logger.error(f"Failed to start enhanced BCI streaming: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start streaming: {str(e)}")


@router.get("/bci/data", response_model=StreamingResponse)
@handle_errors(timeout_seconds=10.0)
@rate_limit(max_requests=60, window_minutes=1)
async def get_enhanced_neural_data(
    request: Request,
    samples: int = Field(default=250, ge=1, le=5000),
    include_quality: bool = Field(default=True),
    bci_bridge: BCIBridge = Depends(get_bci_bridge)
):
    """Enhanced neural data retrieval with quality assessment."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        if len(bci_bridge.data_buffer) == 0:
            return StreamingResponse(
                data=[],
                timestamp=time.time(),
                channels=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get neural data
        buffer_data = bci_bridge.get_buffer(samples)
        
        if buffer_data.size == 0:
            return StreamingResponse(
                data=[],
                timestamp=time.time(),
                channels=[f"CH{i+1}" for i in range(bci_bridge.channels)],
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Calculate signal quality if requested
        signal_quality = {}
        confidence = None
        
        if include_quality and buffer_data.size > 0:
            try:
                import numpy as np
                
                # Basic quality metrics per channel
                for i in range(buffer_data.shape[0]):
                    channel_data = buffer_data[i, :]
                    if len(channel_data) > 0:
                        # Signal-to-noise ratio estimate
                        signal_power = np.var(channel_data)
                        noise_estimate = np.var(np.diff(channel_data))  # High-frequency component as noise proxy
                        snr = signal_power / (noise_estimate + 1e-10)  # Avoid division by zero
                        
                        signal_quality[f"CH{i+1}"] = float(min(snr / 10.0, 1.0))  # Normalize to 0-1
                
                # Overall confidence as mean quality
                confidence = float(np.mean(list(signal_quality.values()))) if signal_quality else 0.0
                
            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")
                confidence = 0.5  # Default confidence
        
        processing_time = (time.time() - start_time) * 1000
        
        return StreamingResponse(
            data=buffer_data.tolist(),
            timestamp=time.time(),
            channels=[f"CH{i+1}" for i in range(bci_bridge.channels)],
            confidence=confidence,
            signal_quality=signal_quality,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Failed to get enhanced neural data: {e}")
        raise HTTPException(status_code=500, detail=f"Data retrieval failed: {str(e)}")


# Enhanced Claude integration with safety monitoring
@router.post("/claude/execute")
@handle_errors(timeout_seconds=20.0)
@rate_limit(max_requests=30, window_minutes=1)
async def execute_enhanced_neural_command(
    request: Request,
    intention_request: IntentionRequest,
    claude_adapter: ClaudeFlowAdapter = Depends(get_claude_adapter),
    validator: InputValidator = Depends(get_input_validator)
):
    """Enhanced neural command execution with comprehensive safety monitoring."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate input
        validated_command = validator.validate_string_input(
            intention_request.command, 
            "neural_command"
        )
        
        # Create intention object
        intention = DecodedIntention(
            command=validated_command,
            confidence=intention_request.confidence,
            context={**intention_request.context, "request_id": request_id},
            timestamp=time.time()
        )
        
        # Execute through Claude with timeout based on priority
        timeout = intention_request.timeout_seconds
        if intention_request.priority == "critical":
            timeout = min(timeout, 5.0)  # Critical operations get faster timeout
        elif intention_request.priority == "low":
            timeout = min(timeout, 30.0)
        
        response = await asyncio.wait_for(
            claude_adapter.execute(intention), 
            timeout=timeout
        )
        
        # Update metrics
        metrics["claude_executions_total"] += 1
        
        # Safety flag monitoring
        if response.safety_flags:
            logger.warning(f"Safety flags detected: {response.safety_flags}")
            
            # Log critical safety issues
            if any(flag in ["emergency", "critical", "urgent_attention"] for flag in response.safety_flags):
                if _SECURITY_AVAILABLE:
                    security_logger.log_suspicious_activity(
                        activity_type="critical_safety_flag",
                        details={
                            "safety_flags": response.safety_flags,
                            "command": validated_command,
                            "request_id": request_id
                        },
                        risk_score=9
                    )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "response": response.content,
            "reasoning": response.reasoning,
            "confidence": response.confidence,
            "safety_flags": response.safety_flags,
            "processing_time_ms": processing_time,
            "tokens_used": response.tokens_used,
            "request_id": request_id,
            "priority": intention_request.priority,
            "timestamp": time.time()
        }
        
    except asyncio.TimeoutError:
        logger.error(f"Claude execution timed out for request {request_id}")
        raise HTTPException(status_code=408, detail=f"Processing timed out after {intention_request.timeout_seconds}s")
    
    except ValidationError as e:
        logger.error(f"Input validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(e)}")
    
    except Exception as e:
        logger.error(f"Enhanced Claude execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


# Enhanced calibration with progress tracking
@router.post("/calibrate")
@handle_errors(timeout_seconds=120.0)
@rate_limit(max_requests=5, window_minutes=10)
async def enhanced_calibration(
    request: Request,
    calibration_request: CalibrationRequest,
    background_tasks: BackgroundTasks,
    bci_bridge: BCIBridge = Depends(get_bci_bridge),
    validator: InputValidator = Depends(get_input_validator)
):
    """Enhanced calibration with progress tracking and validation."""
    calibration_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate subject ID
        validated_subject_id = validator.validate_string_input(
            calibration_request.subject_id,
            "subject_id"
        )
        
        # Enhanced calibration process
        async def calibration_process():
            """Background calibration with progress tracking."""
            try:
                logger.info(f"Starting calibration {calibration_id} for subject {validated_subject_id}")
                
                # Simulate calibration process with progress tracking
                # In real implementation, this would interface with actual calibration logic
                await asyncio.sleep(0.5)  # Simulate calibration time
                
                # Run actual calibration
                bci_bridge.calibrate()
                
                # Log successful calibration
                if _SECURITY_AVAILABLE:
                    security_logger.log_security_event(
                        event_type=SecurityEvent.CONFIGURATION_CHANGE,
                        resource="bci_calibration",
                        action="complete",
                        details={
                            "calibration_id": calibration_id,
                            "subject_id": validated_subject_id,
                            "paradigm": calibration_request.paradigm,
                            "trials": calibration_request.trials
                        },
                        risk_score=2
                    )
                
                logger.info(f"Calibration {calibration_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Calibration process failed: {e}")
                if _SECURITY_AVAILABLE:
                    security_logger.log_system_error(
                        component="bci_calibration",
                        error_type="calibration_failure",
                        error_message=str(e)
                    )
        
        # Start calibration in background
        background_tasks.add_task(calibration_process)
        
        return {
            "message": f"Enhanced calibration started for {calibration_request.paradigm}",
            "calibration_id": calibration_id,
            "subject_id": validated_subject_id,
            "paradigm": calibration_request.paradigm,
            "trials": calibration_request.trials,
            "status": "in_progress",
            "estimated_duration_seconds": calibration_request.trials * 2.0,  # Estimate
            "timestamp": time.time(),
            "baseline_duration": calibration_request.baseline_duration
        }
        
    except ValidationError as e:
        logger.error(f"Calibration validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid calibration request: {str(e)}")
    
    except Exception as e:
        logger.error(f"Enhanced calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


# Comprehensive metrics endpoint
@router.get("/metrics", response_model=MetricsResponse)
@handle_errors(timeout_seconds=10.0)
@rate_limit(max_requests=100, window_minutes=1)
async def get_comprehensive_metrics(
    request: Request,
    format: str = Field(default="json", regex="^(json|prometheus)$"),
    bci_bridge: BCIBridge = Depends(get_bci_bridge),
    claude_adapter: ClaudeFlowAdapter = Depends(get_claude_adapter)
):
    """Comprehensive system metrics with multiple output formats."""
    collection_start = time.time()
    
    try:
        # Collect system metrics
        device_info = bci_bridge.get_device_info()
        health_status = bci_bridge.get_health_status()
        
        system_metrics = {
            "bci_connected": 1 if device_info.get("connected") else 0,
            "bci_streaming": 1 if device_info.get("streaming") else 0,
            "buffer_size": len(bci_bridge.data_buffer),
            "buffer_utilization_pct": (len(bci_bridge.data_buffer) / bci_bridge.buffer_size) * 100,
            "channels": device_info.get("channels", 0),
            "sampling_rate": device_info.get("sampling_rate", 0),
            "uptime_seconds": time.time() - health_status.get("timestamp", time.time())
        }
        
        # Performance metrics
        performance_metrics = {
            "requests_total": metrics["requests_total"],
            "requests_success": metrics["requests_success"],
            "requests_error": metrics["requests_error"],
            "success_rate_pct": (metrics["requests_success"] / max(metrics["requests_total"], 1)) * 100,
            "avg_response_time_ms": metrics["avg_response_time_ms"]
        }
        
        # Neural processing metrics
        neural_metrics = {
            "neural_processing_total": metrics["neural_processing_total"],
            "claude_executions_total": metrics["claude_executions_total"],
            "conversation_history_length": len(claude_adapter.get_conversation_history()),
            "error_count": health_status.get("error_count", 0)
        }
        
        collection_time = (time.time() - collection_start) * 1000
        
        if format == "prometheus":
            # Format as Prometheus metrics
            prometheus_lines = []
            
            for metric_group, metrics_dict in [
                ("system", system_metrics),
                ("performance", performance_metrics), 
                ("neural", neural_metrics)
            ]:
                for metric_name, value in metrics_dict.items():
                    prometheus_lines.append(f"bci_{metric_group}_{metric_name} {value}")
            
            prometheus_lines.append(f"bci_metrics_collection_duration_ms {collection_time}")
            
            return Response(
                content="\n".join(prometheus_lines),
                media_type="text/plain"
            )
        
        else:  # JSON format
            return MetricsResponse(
                system_metrics=system_metrics,
                performance_metrics=performance_metrics,
                neural_metrics=neural_metrics,
                timestamp=time.time(),
                collection_duration_ms=collection_time
            )
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


# Enhanced WebSocket with security and quality monitoring  
@router.websocket("/ws/stream")
async def enhanced_websocket_stream(
    websocket: WebSocket,
    quality_threshold: float = 0.3,
    max_fps: int = 30
):
    """Enhanced WebSocket endpoint with quality filtering and rate limiting."""
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    logger.info(f"WebSocket connection established: {connection_id}")
    
    try:
        # Get BCI bridge from app state
        bci_bridge = websocket.app.state.bci_bridge
        if not bci_bridge:
            await websocket.close(code=1003, reason="BCI service unavailable")
            return
        
        frame_interval = 1.0 / max_fps
        last_frame_time = 0
        
        async for neural_data in bci_bridge.stream():
            try:
                current_time = time.time()
                
                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    continue
                
                # Basic quality assessment
                signal_quality = 1.0  # Default quality
                try:
                    if neural_data.data.size > 0:
                        signal_variance = float(neural_data.data.var())
                        signal_quality = min(signal_variance / 100.0, 1.0)  # Normalize variance
                except Exception:
                    pass
                
                # Quality filtering
                if signal_quality < quality_threshold:
                    continue
                
                # Prepare data packet
                data_packet = {
                    "timestamp": neural_data.timestamp,
                    "data": neural_data.data.tolist() if neural_data.data.size < 1000 else [],  # Limit data size
                    "channels": neural_data.channels,
                    "sampling_rate": neural_data.sampling_rate,
                    "quality": signal_quality,
                    "connection_id": connection_id
                }
                
                await websocket.send_json(data_packet)
                last_frame_time = current_time
                
            except Exception as e:
                logger.error(f"WebSocket data processing error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}")
    finally:
        logger.info(f"WebSocket connection closed: {connection_id}")
        if not websocket.client_state.DISCONNECTED:
            await websocket.close()


# Real-time processing pipeline with enhanced monitoring
@router.post("/realtime/process")
@handle_errors(timeout_seconds=30.0)
@rate_limit(max_requests=5, window_minutes=1)
async def enhanced_realtime_processing(
    request: Request,
    background_tasks: BackgroundTasks,
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0),
    max_processing_time: float = Field(default=15.0, ge=1.0, le=30.0),
    bci_bridge: BCIBridge = Depends(get_bci_bridge),
    claude_adapter: ClaudeFlowAdapter = Depends(get_claude_adapter)
):
    """Enhanced real-time processing with adaptive thresholds and monitoring."""
    session_id = str(uuid.uuid4())
    
    try:
        if bci_bridge.is_streaming:
            return {
                "message": "Real-time processing already active", 
                "status": "active",
                "session_id": session_id
            }
        
        async def enhanced_processing_pipeline():
            """Enhanced processing pipeline with adaptive behavior."""
            processing_count = 0
            error_count = 0
            adaptive_threshold = confidence_threshold
            start_time = time.time()
            
            try:
                async for neural_data in bci_bridge.stream():
                    if time.time() - start_time > max_processing_time:
                        logger.info(f"Processing session {session_id} completed (time limit)")
                        break
                    
                    try:
                        # Decode intention
                        intention = bci_bridge.decode_intention(neural_data)
                        
                        # Adaptive confidence threshold
                        if processing_count > 10:
                            # Increase threshold if too many low-quality predictions
                            avg_confidence = getattr(enhanced_processing_pipeline, 'avg_confidence', 0.5)
                            if avg_confidence < 0.4:
                                adaptive_threshold = min(confidence_threshold + 0.1, 0.9)
                            else:
                                adaptive_threshold = confidence_threshold
                        
                        # Process high-confidence intentions
                        if intention.confidence > adaptive_threshold:
                            response = await claude_adapter.execute(intention)
                            processing_count += 1
                            
                            # Update running average confidence
                            current_avg = getattr(enhanced_processing_pipeline, 'avg_confidence', 0.5)
                            enhanced_processing_pipeline.avg_confidence = (
                                current_avg * 0.9 + intention.confidence * 0.1
                            )
                            
                            logger.info(
                                f"Processed intention {processing_count}: "
                                f"{intention.command} -> {response.content[:100]}..."
                            )
                            
                            # Safety monitoring
                            if response.safety_flags and any(
                                flag in ["emergency", "critical"] for flag in response.safety_flags
                            ):
                                logger.critical(f"Critical safety alert in session {session_id}")
                                if _SECURITY_AVAILABLE:
                                    security_logger.log_suspicious_activity(
                                        activity_type="critical_safety_alert",
                                        details={
                                            "session_id": session_id,
                                            "safety_flags": response.safety_flags,
                                            "command": intention.command
                                        },
                                        risk_score=10
                                    )
                        
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Processing error in session {session_id}: {e}")
                        
                        if error_count > 5:  # Stop after too many errors
                            logger.error(f"Too many errors in session {session_id}, stopping")
                            break
                            
                        continue
            
            except Exception as e:
                logger.error(f"Processing pipeline failed for session {session_id}: {e}")
            finally:
                bci_bridge.stop_streaming()
                logger.info(
                    f"Processing session {session_id} ended: "
                    f"{processing_count} processed, {error_count} errors"
                )
        
        background_tasks.add_task(enhanced_processing_pipeline)
        
        return {
            "message": "Enhanced real-time processing started",
            "status": "active",
            "session_id": session_id,
            "confidence_threshold": confidence_threshold,
            "max_processing_time": max_processing_time,
            "timestamp": time.time(),
            "features": ["adaptive_thresholds", "safety_monitoring", "error_recovery"]
        }
        
    except Exception as e:
        logger.error(f"Failed to start enhanced real-time processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")