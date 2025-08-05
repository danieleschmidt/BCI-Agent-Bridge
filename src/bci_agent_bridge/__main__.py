"""
Main entry point for BCI-Agent-Bridge application.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .api import create_app
from .core.bridge import BCIBridge
from .adapters.claude_flow import ClaudeFlowAdapter


# Global application state
app_state = {
    "bci_bridge": None,
    "claude_adapter": None,
    "running": False
}


def setup_logging() -> None:
    """Setup application logging."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "standard")
    
    if log_format == "json":
        import json
        import datetime
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                if record.exc_info:
                    log_obj["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_obj)
        
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(handler)


def initialize_bci_components() -> tuple[BCIBridge, ClaudeFlowAdapter]:
    """Initialize BCI bridge and Claude adapter."""
    logger = logging.getLogger(__name__)
    
    # BCI Bridge configuration
    device = os.getenv("BCI_DEVICE", "Simulation")
    channels = int(os.getenv("BCI_CHANNELS", "8"))
    sampling_rate = int(os.getenv("BCI_SAMPLING_RATE", "250"))
    paradigm = os.getenv("BCI_PARADIGM", "P300")
    
    logger.info(f"Initializing BCI Bridge: {device}, {channels} channels, {sampling_rate}Hz, {paradigm}")
    
    bci_bridge = BCIBridge(
        device=device,
        channels=channels,
        sampling_rate=sampling_rate,
        paradigm=paradigm,
        privacy_mode=True
    )
    
    # Claude adapter configuration
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is required")
        sys.exit(1)
    
    safety_mode = os.getenv("PRIVACY_MODE", "medical")
    
    logger.info(f"Initializing Claude adapter with {safety_mode} safety mode")
    
    claude_adapter = ClaudeFlowAdapter(
        api_key=api_key,
        safety_mode=safety_mode,
        model="claude-3-sonnet-20240229"
    )
    
    return bci_bridge, claude_adapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger = logging.getLogger(__name__)
    
    # Startup
    logger.info("Starting BCI-Agent-Bridge application")
    
    try:
        # Initialize components
        bci_bridge, claude_adapter = initialize_bci_components()
        app_state["bci_bridge"] = bci_bridge
        app_state["claude_adapter"] = claude_adapter
        app_state["running"] = True
        
        # Store in app state for API access
        app.state.bci_bridge = bci_bridge
        app.state.claude_adapter = claude_adapter
        
        logger.info("BCI-Agent-Bridge startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start BCI-Agent-Bridge: {e}")
        sys.exit(1)
    
    finally:
        # Shutdown
        logger.info("Shutting down BCI-Agent-Bridge application")
        app_state["running"] = False
        
        if app_state["bci_bridge"]:
            app_state["bci_bridge"].stop_streaming()
        
        logger.info("BCI-Agent-Bridge shutdown completed")


def create_application() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="BCI-Agent-Bridge",
        description="Real-time Brain-Computer Interface to LLM bridge",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    from .api.routes import router
    app.include_router(router, prefix="/api/v1")
    
    return app


def handle_signal(signum: int, frame) -> None:
    """Handle shutdown signals."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    app_state["running"] = False


def main() -> None:
    """Main application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Create application
    app = create_application()
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("WORKER_PROCESSES", "1"))
    
    logger.info(f"Starting BCI-Agent-Bridge server on {host}:{port}")
    
    # Production vs development configuration
    if os.getenv("ENVIRONMENT") == "production":
        uvicorn.run(
            "bci_agent_bridge.__main__:create_application",
            factory=True,
            host=host,
            port=port,
            workers=workers,
            log_config=None,  # Use our custom logging
            access_log=False
        )
    else:
        # Development mode
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_config=None,
            access_log=True,
            reload=False  # Disable reload due to custom initialization
        )


if __name__ == "__main__":
    main()