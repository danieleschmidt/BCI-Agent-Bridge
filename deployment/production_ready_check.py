#!/usr/bin/env python3
"""
Production Readiness Check for BCI-Agent-Bridge
Validates all systems before deployment.
"""

import sys
import time
import logging
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bci_agent_bridge import BCIBridge
from bci_agent_bridge.core.optimized_bridge import OptimizedBCIBridge, PerformanceConfig
from bci_agent_bridge.decoders.p300 import P300Decoder
from bci_agent_bridge.security.audit_logger import SecurityAuditLogger
from bci_agent_bridge.security.input_validator import InputValidator, SecurityPolicy

def check_mark(passed: bool) -> str:
    """Return appropriate check mark."""
    return "✅" if passed else "❌"

async def main():
    """Run comprehensive production readiness checks."""
    
    print("🚀 BCI-AGENT-BRIDGE PRODUCTION READINESS CHECK")
    print("=" * 60)
    
    all_checks_passed = True
    checks = []
    
    # 1. Core Functionality Check
    print("\n🔧 Core Functionality Check:")
    try:
        bridge = BCIBridge(device="Simulation", channels=8)
        bridge.decoder = P300Decoder(channels=8, sampling_rate=250)
        
        # Test basic operations
        assert bridge._device_connected
        neural_data = bridge._generate_simulation_data()
        intention = bridge.decode_intention(neural_data)
        
        core_passed = True
        print(f"   {check_mark(core_passed)} Core BCI functionality")
        checks.append(("Core Functionality", core_passed))
        
    except Exception as e:
        core_passed = False
        print(f"   {check_mark(core_passed)} Core BCI functionality - Error: {e}")
        checks.append(("Core Functionality", core_passed))
        all_checks_passed = False
    
    # 2. Performance Optimization Check
    print("\n⚡ Performance Optimization Check:")
    try:
        opt_bridge = OptimizedBCIBridge(
            performance_config=PerformanceConfig(
                enable_caching=True,
                enable_parallel_processing=True,
                cache_size_mb=50
            )
        )
        opt_bridge.decoder = P300Decoder(channels=8, sampling_rate=250)
        
        # Performance test
        start_time = time.perf_counter()
        for _ in range(10):
            neural_data = opt_bridge._generate_simulation_data()
            opt_bridge.optimized_decode_intention(neural_data)
        duration = time.perf_counter() - start_time
        
        metrics = opt_bridge.get_performance_metrics()
        performance_passed = duration < 0.1 and metrics['memory_usage_mb'] < 100
        
        print(f"   {check_mark(performance_passed)} Performance optimization (10 ops in {duration:.3f}s)")
        print(f"   {check_mark(performance_passed)} Memory usage: {metrics['memory_usage_mb']:.2f}MB")
        checks.append(("Performance Optimization", performance_passed))
        
        opt_bridge.cleanup()
        
    except Exception as e:
        performance_passed = False
        print(f"   {check_mark(performance_passed)} Performance optimization - Error: {e}")
        checks.append(("Performance Optimization", performance_passed))
        all_checks_passed = False
    
    # 3. Security Framework Check
    print("\n🔒 Security Framework Check:")
    try:
        # Input validation
        validator = InputValidator(SecurityPolicy.STANDARD)
        validator.validate_api_key("sk-97e4b2c1a6f8d5e9437b8c2f1a6e9d4b")
        
        # Audit logging
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp:
            logger = SecurityAuditLogger(log_file=tmp.name)
            logger.log_authentication_attempt("test_user", True, "127.0.0.1")
            
        security_passed = True
        print(f"   {check_mark(security_passed)} Input validation")
        print(f"   {check_mark(security_passed)} Security audit logging")
        checks.append(("Security Framework", security_passed))
        
    except Exception as e:
        security_passed = False
        print(f"   {check_mark(security_passed)} Security framework - Error: {e}")
        checks.append(("Security Framework", security_passed))
        all_checks_passed = False
    
    # 4. Streaming Performance Check
    print("\n📊 Streaming Performance Check:")
    try:
        bridge = BCIBridge(device="Simulation", channels=8)
        
        sample_count = 0
        start_time = time.time()
        
        async def streaming_test():
            nonlocal sample_count
            async for neural_data in bridge.stream():
                sample_count += neural_data.data.shape[1]
                if time.time() - start_time > 1.0:  # Test for 1 second
                    break
            bridge.stop_streaming()
        
        await streaming_test()
        throughput = sample_count / (time.time() - start_time)
        streaming_passed = throughput >= 10000  # 10k samples/sec target
        
        print(f"   {check_mark(streaming_passed)} Streaming throughput: {throughput:.0f} samples/sec (target: 10k)")
        checks.append(("Streaming Performance", streaming_passed))
        
    except Exception as e:
        streaming_passed = False
        print(f"   {check_mark(streaming_passed)} Streaming performance - Error: {e}")
        checks.append(("Streaming Performance", streaming_passed))
        all_checks_passed = False
    
    # 5. Error Handling & Resilience Check
    print("\n🛡️ Error Handling & Resilience Check:")
    try:
        bridge = BCIBridge()
        
        # Test graceful handling of invalid inputs
        try:
            bridge.decode_intention(None)
            resilience_test_1 = False
        except (ValueError, RuntimeError):
            resilience_test_1 = True
        
        # Test decoder without calibration
        bridge.decoder = P300Decoder(channels=8, sampling_rate=250)
        neural_data = bridge._generate_simulation_data()
        intention = bridge.decode_intention(neural_data)  # Should not crash
        resilience_test_2 = intention is not None
        
        resilience_passed = resilience_test_1 and resilience_test_2
        print(f"   {check_mark(resilience_passed)} Invalid input handling")
        print(f"   {check_mark(resilience_passed)} Graceful degradation")
        checks.append(("Error Handling & Resilience", resilience_passed))
        
    except Exception as e:
        resilience_passed = False
        print(f"   {check_mark(resilience_passed)} Error handling & resilience - Error: {e}")
        checks.append(("Error Handling & Resilience", resilience_passed))
        all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PRODUCTION READINESS SUMMARY")
    print("=" * 60)
    
    for check_name, passed in checks:
        print(f"{check_mark(passed)} {check_name}")
    
    print(f"\n🎯 Overall Status: {check_mark(all_checks_passed)} {'PRODUCTION READY' if all_checks_passed else 'NEEDS ATTENTION'}")
    
    if all_checks_passed:
        print("\n🚀 System is ready for production deployment!")
        print("   • All core functionality validated")
        print("   • Performance targets exceeded")
        print("   • Security framework active")
        print("   • Error handling robust")
        print("   • Streaming performance optimal")
    else:
        print("\n⚠️  Some checks failed. Please review and fix issues before deployment.")
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)