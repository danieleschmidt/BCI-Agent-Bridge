"""
API routes for BCI-Agent-Bridge.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import time
import asyncio
import logging

from ..core.bridge import BCIBridge, NeuralData, DecodedIntention
from ..adapters.claude_flow import ClaudeFlowAdapter, ClaudeResponse

router = APIRouter()
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str = "0.1.0"
    components: Dict[str, str]


class NeuralDataRequest(BaseModel):
    channels: int
    sampling_rate: int
    paradigm: str
    duration_seconds: float = 1.0


class IntentionRequest(BaseModel):
    command: str
    confidence: float
    context: Dict[str, Any]


class StreamingResponse(BaseModel):
    data: List[float]
    timestamp: float
    channels: List[str]
    confidence: Optional[float] = None


# Dependency injection
async def get_bci_bridge(request: Request) -> BCIBridge:
    """Get BCI bridge from application state."""
    if not hasattr(request.app.state, 'bci_bridge') or request.app.state.bci_bridge is None:
        raise HTTPException(status_code=503, detail="BCI Bridge not initialized")
    return request.app.state.bci_bridge


async def get_claude_adapter(request: Request) -> ClaudeFlowAdapter:
    """Get Claude adapter from application state."""
    if not hasattr(request.app.state, 'claude_adapter') or request.app.state.claude_adapter is None:
        raise HTTPException(status_code=503, detail="Claude adapter not initialized")
    return request.app.state.claude_adapter


# Health and status endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check(
    request: Request,
    bci_bridge: BCIBridge = Depends(get_bci_bridge),
    claude_adapter: ClaudeFlowAdapter = Depends(get_claude_adapter)
):
    """Health check endpoint."""
    try:
        device_info = bci_bridge.get_device_info()
        
        components = {
            "bci_bridge": "healthy" if device_info["connected"] else "degraded",
            "claude_adapter": "healthy",
            "database": "healthy",  # TODO: Add actual database check
            "privacy": "healthy"
        }
        
        return HealthResponse(
            status="healthy" if all(status == "healthy" for status in components.values()) else "degraded",
            timestamp=time.time(),
            components=components
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@router.get("/status")
async def get_system_status(
    bci_bridge: BCIBridge = Depends(get_bci_bridge),
    claude_adapter: ClaudeFlowAdapter = Depends(get_claude_adapter)
):
    """Get detailed system status."""
    try:
        device_info = bci_bridge.get_device_info()
        privacy_status = {}  # TODO: Add privacy system status
        
        return {
            "bci_system": device_info,
            "streaming": {
                "active": device_info["streaming"],
                "buffer_size": len(bci_bridge.data_buffer)
            },
            "privacy": privacy_status,
            "claude_adapter": {
                "model": "claude-3-sonnet-20240229",
                "safety_mode": "medical"
            },
            "uptime": time.time(),
            "version": "0.1.0"
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# BCI data endpoints
@router.post("/bci/start")
async def start_bci_streaming(
    background_tasks: BackgroundTasks,
    bci_bridge: BCIBridge = Depends(get_bci_bridge)
):
    """Start BCI data streaming."""
    try:
        if bci_bridge.is_streaming:
            return {"message": "BCI streaming already active", "status": "active"}
        
        # Start streaming in background
        async def stream_handler():
            try:
                async for neural_data in bci_bridge.stream():
                    # Process neural data
                    logger.debug(f"Received neural data: {neural_data.timestamp}")
                    # Here you could add real-time processing logic
            except Exception as e:
                logger.error(f"Streaming error: {e}")
        
        background_tasks.add_task(stream_handler)
        
        return {"message": "BCI streaming started", "status": "active"}
        
    except Exception as e:
        logger.error(f"Failed to start BCI streaming: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start streaming: {str(e)}")


@router.post("/bci/stop")
async def stop_bci_streaming(bci_bridge: BCIBridge = Depends(get_bci_bridge)):
    """Stop BCI data streaming."""
    try:
        bci_bridge.stop_streaming()
        return {"message": "BCI streaming stopped", "status": "stopped"}
    except Exception as e:
        logger.error(f"Failed to stop BCI streaming: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop streaming: {str(e)}")


@router.get("/bci/data")
async def get_recent_data(
    samples: int = 250,
    bci_bridge: BCIBridge = Depends(get_bci_bridge)
):
    """Get recent BCI data from buffer."""
    try:
        if len(bci_bridge.data_buffer) == 0:
            return {"data": [], "message": "No data available"}
        
        # Get recent samples
        buffer_data = bci_bridge.get_buffer(samples)
        
        if buffer_data.size == 0:
            return {"data": [], "message": "Insufficient data in buffer"}
        
        return {
            "data": buffer_data.tolist(),
            "shape": buffer_data.shape,
            "timestamp": time.time(),
            "channels": bci_bridge.channels,
            "sampling_rate": bci_bridge.sampling_rate
        }
        
    except Exception as e:
        logger.error(f"Failed to get BCI data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data: {str(e)}")


# Intention decoding endpoints
@router.post("/decode/intention")
async def decode_neural_intention(
    request: NeuralDataRequest,
    bci_bridge: BCIBridge = Depends(get_bci_bridge)
):
    """Decode intention from current neural data."""
    try:
        # Get recent data window
        window_ms = int(request.duration_seconds * 1000)
        neural_data_array = bci_bridge.read_window(window_ms)
        
        if neural_data_array.size == 0:
            raise HTTPException(status_code=400, detail="No neural data available")
        
        # Create NeuralData object
        neural_data = NeuralData(
            data=neural_data_array,
            timestamp=time.time(),
            channels=[f"CH{i+1}" for i in range(request.channels)],
            sampling_rate=request.sampling_rate
        )
        
        # Decode intention
        intention = bci_bridge.decode_intention(neural_data)
        
        return {
            "command": intention.command,
            "confidence": intention.confidence,
            "context": intention.context,
            "timestamp": intention.timestamp,
            "paradigm": request.paradigm
        }
        
    except Exception as e:
        logger.error(f"Intention decoding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")


# Claude integration endpoints
@router.post("/claude/execute")
async def execute_neural_command(
    request: IntentionRequest,
    claude_adapter: ClaudeFlowAdapter = Depends(get_claude_adapter)
):
    """Execute neural intention through Claude."""
    try:
        from ..core.bridge import DecodedIntention
        
        # Create intention object
        intention = DecodedIntention(
            command=request.command,
            confidence=request.confidence,
            context=request.context,
            timestamp=time.time()
        )
        
        # Execute through Claude
        response = await claude_adapter.execute(intention)
        
        return {
            "response": response.content,
            "reasoning": response.reasoning,
            "confidence": response.confidence,
            "safety_flags": response.safety_flags,
            "processing_time_ms": response.processing_time_ms,
            "tokens_used": response.tokens_used
        }
        
    except Exception as e:
        logger.error(f"Claude execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


# Real-time processing endpoint
@router.post("/realtime/process")
async def realtime_bci_processing(
    background_tasks: BackgroundTasks,
    bci_bridge: BCIBridge = Depends(get_bci_bridge),
    claude_adapter: ClaudeFlowAdapter = Depends(get_claude_adapter)
):
    """Start real-time BCI processing pipeline."""
    try:
        if bci_bridge.is_streaming:
            return {"message": "Real-time processing already active", "status": "active"}
        
        async def processing_pipeline():
            """Real-time processing pipeline."""
            async for neural_data in bci_bridge.stream():
                try:
                    # Decode intention
                    intention = bci_bridge.decode_intention(neural_data)
                    
                    # Only process high-confidence intentions
                    if intention.confidence > 0.7:
                        # Execute through Claude
                        response = await claude_adapter.execute(intention)
                        
                        logger.info(f"Processed intention: {intention.command} -> {response.content}")
                        
                        # Here you could emit WebSocket events, store results, etc.
                        
                except Exception as e:
                    logger.error(f"Processing pipeline error: {e}")
                    continue
        
        background_tasks.add_task(processing_pipeline)
        
        return {"message": "Real-time processing started", "status": "active"}
        
    except Exception as e:
        logger.error(f"Failed to start real-time processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# Calibration endpoints
@router.post("/calibrate")
async def calibrate_decoder(
    paradigm: str = "P300",
    trials: int = 50,
    bci_bridge: BCIBridge = Depends(get_bci_bridge)
):
    """Calibrate BCI decoder."""
    try:
        # Start calibration
        bci_bridge.calibrate()
        
        return {
            "message": f"Calibration completed for {paradigm}",
            "paradigm": paradigm,
            "trials": trials,
            "status": "calibrated",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


# Metrics endpoint for monitoring
@router.get("/metrics")
async def get_metrics(
    bci_bridge: BCIBridge = Depends(get_bci_bridge),
    claude_adapter: ClaudeFlowAdapter = Depends(get_claude_adapter)
):
    """Get system metrics for monitoring."""
    try:
        device_info = bci_bridge.get_device_info()
        
        metrics = {
            "bci_connected": 1 if device_info["connected"] else 0,
            "bci_streaming": 1 if device_info["streaming"] else 0,
            "buffer_size": len(bci_bridge.data_buffer),
            "channels": device_info["channels"],
            "sampling_rate": device_info["sampling_rate"],
            "uptime": time.time(),
            "conversation_history_length": len(claude_adapter.get_conversation_history())
        }
        
        # Format as Prometheus metrics
        prometheus_metrics = []
        for key, value in metrics.items():
            prometheus_metrics.append(f"bci_{key} {value}")
        
        return JSONResponse(
            content="\n".join(prometheus_metrics),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics failed: {str(e)}")


# WebSocket endpoint for real-time data
@router.websocket("/ws/stream")
async def websocket_stream(websocket, bci_bridge: BCIBridge = Depends(get_bci_bridge)):
    """WebSocket endpoint for real-time neural data streaming."""
    await websocket.accept()
    
    try:
        async for neural_data in bci_bridge.stream():
            # Send data to WebSocket client
            data = {
                "timestamp": neural_data.timestamp,
                "data": neural_data.data.tolist(),
                "channels": neural_data.channels,
                "sampling_rate": neural_data.sampling_rate
            }
            
            await websocket.send_json(data)
            
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}")
        await websocket.close(code=1000)