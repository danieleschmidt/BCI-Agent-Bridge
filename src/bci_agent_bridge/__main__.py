#!/usr/bin/env python3
"""
Command-line interface and main entry point for BCI-Agent-Bridge.
Provides both CLI demo mode and API server mode.
"""

import asyncio
import argparse
import logging
import os
import signal
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

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
    try:
        from .api.routes import router
        app.include_router(router, prefix="/api/v1")
    except ImportError:
        logger.warning("API routes not available - running in CLI-only mode")
    
    return app


def handle_signal(signum: int, frame) -> None:
    """Handle shutdown signals."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    app_state["running"] = False


async def run_demo(args: argparse.Namespace) -> None:
    """Run a demonstration of the BCI bridge."""
    logger = logging.getLogger(__name__)
    
    print("🧠 BCI-Agent-Bridge Demo Starting...")
    print(f"Device: {args.device}")
    print(f"Paradigm: {args.paradigm}")
    print(f"Channels: {args.channels}")
    print(f"Sampling Rate: {args.sampling_rate} Hz")
    print("-" * 50)
    
    # Initialize BCI Bridge
    bridge = BCIBridge(
        device=args.device,
        channels=args.channels,
        sampling_rate=args.sampling_rate,
        paradigm=args.paradigm,
        privacy_mode=not args.disable_privacy
    )
    
    # Initialize Claude adapter if API key provided
    claude_adapter = None
    if args.claude_api_key:
        claude_adapter = ClaudeFlowAdapter(
            api_key=args.claude_api_key,
            safety_mode=args.safety_mode
        )
        print("✅ Claude Flow adapter initialized")
    
    print("🔄 Starting neural data stream...")
    
    try:
        sample_count = 0
        async for neural_data in bridge.stream():
            sample_count += 1
            
            # Decode intention from neural data
            intention = bridge.decode_intention(neural_data)
            
            print(f"\n📡 Sample {sample_count}")
            print(f"   Command: {intention.command}")
            print(f"   Confidence: {intention.confidence:.3f}")
            print(f"   Timestamp: {intention.timestamp:.3f}")
            
            # Process through Claude if available and confidence is high
            if claude_adapter and intention.confidence > 0.7:
                print("   🤖 Processing through Claude...")
                try:
                    claude_response = await claude_adapter.execute(intention)
                    print(f"   Claude Response: {claude_response.content}")
                    print(f"   Processing Time: {claude_response.processing_time_ms:.1f}ms")
                    
                    if claude_response.safety_flags:
                        print(f"   ⚠️  Safety Flags: {', '.join(claude_response.safety_flags)}")
                        
                except Exception as e:
                    print(f"   ❌ Claude processing error: {e}")
            
            # Stop after specified number of samples
            if sample_count >= args.samples:
                bridge.stop_streaming()
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"❌ Error: {e}")
    finally:
        bridge.stop_streaming()
        print("\n✅ Demo completed")


def run_interactive_mode(args: argparse.Namespace) -> None:
    """Run interactive command mode."""
    print("🧠 BCI-Agent-Bridge Interactive Mode")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 40)
    
    bridge = BCIBridge(
        device=args.device,
        channels=args.channels,
        sampling_rate=args.sampling_rate,
        paradigm=args.paradigm
    )
    
    while True:
        try:
            command = input("\nbci> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif command == 'help':
                print("""
Available commands:
  info        - Show device information
  calibrate   - Run calibration
  test        - Generate test data
  status      - Show system status
  help        - Show this help
  quit        - Exit interactive mode
""")
            elif command == 'info':
                info = bridge.get_device_info()
                print(json.dumps(info, indent=2))
            elif command == 'calibrate':
                print("🔧 Running calibration...")
                bridge.calibrate()
                print("✅ Calibration completed")
            elif command == 'test':
                print("🧪 Generating test neural data...")
                test_data = bridge._generate_simulation_data()
                print(f"Generated data shape: {test_data.shape}")
            elif command == 'status':
                print(f"Device: {bridge.device.value}")
                print(f"Streaming: {bridge.is_streaming}")
                print(f"Buffer size: {len(bridge.data_buffer)}")
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main() -> None:
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="BCI-Agent-Bridge: Neural-to-Language Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with P300 paradigm
  python -m bci_agent_bridge --paradigm P300 --samples 10
  
  # Interactive mode with motor imagery
  python -m bci_agent_bridge --interactive --paradigm MotorImagery
  
  # Full demo with Claude integration
  python -m bci_agent_bridge --claude-api-key YOUR_KEY --safety-mode medical
  
  # Start API server
  python -m bci_agent_bridge --server
"""
    )
    
    # Device configuration
    parser.add_argument('--device', default='Simulation',
                       choices=['Simulation', 'OpenBCI', 'Emotiv', 'NeuroSky', 'Muse'],
                       help='BCI device type')
    parser.add_argument('--channels', type=int, default=8,
                       help='Number of EEG channels')
    parser.add_argument('--sampling-rate', type=int, default=250,
                       help='Sampling rate in Hz')
    parser.add_argument('--paradigm', default='P300',
                       choices=['P300', 'MotorImagery', 'SSVEP'],
                       help='BCI paradigm')
    
    # Demo configuration
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples to process in demo')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--server', action='store_true',
                       help='Start API server mode')
    
    # Claude integration
    parser.add_argument('--claude-api-key', type=str,
                       help='Claude API key for LLM processing')
    parser.add_argument('--safety-mode', default='medical',
                       choices=['medical', 'standard', 'research'],
                       help='Safety mode for Claude processing')
    
    # Privacy and logging
    parser.add_argument('--disable-privacy', action='store_true',
                       help='Disable privacy mode (not recommended)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging with CLI-friendly format
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if args.server else '%(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bci_bridge.log')
        ]
    )
    
    # Validate API key requirement for Claude features
    if not args.claude_api_key and not args.interactive and not args.server:
        print("⚠️  No Claude API key provided. Running without LLM processing.")
        print("   Set --claude-api-key to enable full functionality.")
    
    try:
        if args.server:
            # Run API server
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
        elif args.interactive:
            run_interactive_mode(args)
        else:
            asyncio.run(run_demo(args))
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()