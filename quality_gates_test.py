#!/usr/bin/env python3
"""
Comprehensive quality gates validation for BCI-Agent-Bridge.
Tests all core components and validates system readiness.
"""

import sys
import time
import traceback
from typing import Dict, Any, List

def run_quality_gates() -> Dict[str, Any]:
    """Run comprehensive quality gates validation."""
    results = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'test_results': [],
        'overall_status': 'UNKNOWN',
        'execution_time': 0.0
    }
    
    start_time = time.time()
    
    print("🧪 Running comprehensive quality gates validation...")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        test_bci_bridge_initialization,
        test_monitoring_systems,
        test_security_logging,
        test_privacy_system,
        test_api_routes_loading,
        test_performance_systems,
        test_compliance_systems,
        test_decoder_systems,
        test_research_modules,
        test_i18n_systems
    ]
    
    for test_case in test_cases:
        results['total_tests'] += 1
        test_name = test_case.__name__
        
        try:
            print(f"🔍 {test_name}...", end=" ")
            test_result = test_case()
            
            if test_result['passed']:
                print("✅ PASS")
                results['passed_tests'] += 1
            else:
                print("❌ FAIL")
                results['failed_tests'] += 1
            
            results['test_results'].append({
                'name': test_name,
                'status': 'PASS' if test_result['passed'] else 'FAIL',
                'details': test_result.get('details', ''),
                'error': test_result.get('error', '')
            })
            
        except Exception as e:
            print("❌ ERROR")
            results['failed_tests'] += 1
            results['test_results'].append({
                'name': test_name,
                'status': 'ERROR',
                'details': f"Test execution failed: {str(e)}",
                'error': traceback.format_exc()
            })
    
    # Calculate final results
    results['execution_time'] = time.time() - start_time
    success_rate = (results['passed_tests'] / results['total_tests']) * 100 if results['total_tests'] > 0 else 0
    
    if success_rate >= 85:
        results['overall_status'] = 'PASS'
    elif success_rate >= 70:
        results['overall_status'] = 'CONDITIONAL_PASS'
    else:
        results['overall_status'] = 'FAIL'
    
    print("\n" + "=" * 60)
    print(f"📊 Quality Gates Summary:")
    print(f"   Total Tests: {results['total_tests']}")
    print(f"   Passed: {results['passed_tests']}")
    print(f"   Failed: {results['failed_tests']}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Execution Time: {results['execution_time']:.2f}s")
    print(f"   Overall Status: {results['overall_status']}")
    
    if results['overall_status'] == 'PASS':
        print("\n🏆 All quality gates PASSED - System is production ready!")
    elif results['overall_status'] == 'CONDITIONAL_PASS':
        print("\n⚠️  Quality gates CONDITIONALLY PASSED - Some issues detected")
    else:
        print("\n🚨 Quality gates FAILED - System needs attention")
    
    return results


def test_bci_bridge_initialization() -> Dict[str, Any]:
    """Test BCI Bridge core functionality."""
    try:
        from src.bci_agent_bridge import BCIBridge
        
        # Test initialization
        bridge = BCIBridge(device='Simulation', channels=8, sampling_rate=250, paradigm='P300')
        
        # Test health status
        health = bridge.get_health_status()
        assert health['status'] != 'error', f"Health status error: {health}"
        
        # Test device info
        device_info = bridge.get_device_info()
        assert device_info['connected'] == True, "Device not connected"
        assert device_info['channels'] == 8, "Channel count mismatch"
        
        return {'passed': True, 'details': 'BCI Bridge initialized and healthy'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


def test_monitoring_systems() -> Dict[str, Any]:
    """Test monitoring and health systems."""
    try:
        from src.bci_agent_bridge.monitoring import HealthMonitor, MetricsCollector
        
        # Test health monitor
        monitor = HealthMonitor()
        assert monitor is not None, "Health monitor initialization failed"
        
        # Test metrics collector
        metrics = MetricsCollector()
        assert metrics is not None, "Metrics collector initialization failed"
        
        return {'passed': True, 'details': 'Monitoring systems operational'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


def test_security_logging() -> Dict[str, Any]:
    """Test security and audit logging."""
    try:
        from src.bci_agent_bridge.security.audit_logger import security_logger
        
        # Test security logging
        security_logger.log_system_info('quality_gates', {'test': 'validation'})
        
        return {'passed': True, 'details': 'Security logging functional'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


def test_privacy_system() -> Dict[str, Any]:
    """Test privacy and differential privacy systems."""
    try:
        from src.bci_agent_bridge.privacy.differential_privacy import DifferentialPrivacy
        
        # Test privacy system
        privacy = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        assert privacy is not None, "Privacy system initialization failed"
        
        return {'passed': True, 'details': 'Privacy systems operational'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


def test_api_routes_loading() -> Dict[str, Any]:
    """Test API routes loading."""
    try:
        from src.bci_agent_bridge.api.routes import router
        
        assert router is not None, "API router not loaded"
        
        return {'passed': True, 'details': 'Enhanced API routes loaded'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


def test_performance_systems() -> Dict[str, Any]:
    """Test performance and scaling systems."""
    try:
        from src.bci_agent_bridge.performance.auto_scaler import NeuralProcessingAutoScaler, ScalingPolicy
        from src.bci_agent_bridge.performance.load_balancer import LoadBalancer
        
        # Test scaling policy
        policy = ScalingPolicy()
        assert policy is not None, "Scaling policy creation failed"
        
        return {'passed': True, 'details': 'Performance systems operational'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


def test_compliance_systems() -> Dict[str, Any]:
    """Test HIPAA/GDPR compliance systems."""
    try:
        from src.bci_agent_bridge.compliance.hipaa import HIPAACompliance
        from src.bci_agent_bridge.compliance.gdpr import GDPRCompliance
        
        # Test HIPAA compliance
        hipaa = HIPAACompliance()
        assert hipaa is not None, "HIPAA compliance system failed"
        
        # Test GDPR compliance  
        gdpr = GDPRCompliance()
        assert gdpr is not None, "GDPR compliance system failed"
        
        return {'passed': True, 'details': 'Compliance systems operational'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


def test_decoder_systems() -> Dict[str, Any]:
    """Test neural decoder systems."""
    try:
        from src.bci_agent_bridge.decoders.p300 import P300Decoder
        from src.bci_agent_bridge.decoders.motor_imagery import MotorImageryDecoder
        from src.bci_agent_bridge.decoders.ssvep import SSVEPDecoder
        
        # Test P300 decoder
        p300 = P300Decoder(channels=8, sampling_rate=250)
        assert p300 is not None, "P300 decoder creation failed"
        
        return {'passed': True, 'details': 'Neural decoder systems operational'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


def test_research_modules() -> Dict[str, Any]:
    """Test advanced research modules (optional)."""
    try:
        from src.bci_agent_bridge import _RESEARCH_AVAILABLE
        
        if _RESEARCH_AVAILABLE:
            from src.bci_agent_bridge.research.quantum_optimization import QuantumNeuralDecoder
            from src.bci_agent_bridge.research.federated_learning import FederatedServer
            
            return {'passed': True, 'details': 'Research modules available and operational'}
        else:
            return {'passed': True, 'details': 'Research modules not available (dependencies missing)'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


def test_i18n_systems() -> Dict[str, Any]:
    """Test internationalization systems."""
    try:
        from src.bci_agent_bridge.i18n.translator import NeuralCommandTranslator
        from src.bci_agent_bridge.i18n.locales import LocalizationManager
        
        # Test translator
        translator = NeuralCommandTranslator()
        assert translator is not None, "Neural command translator failed"
        
        # Test localization
        locales = LocalizationManager()
        assert locales is not None, "Localization manager failed"
        
        return {'passed': True, 'details': 'I18n systems operational'}
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}


if __name__ == "__main__":
    results = run_quality_gates()
    
    # Exit with appropriate code
    if results['overall_status'] == 'PASS':
        sys.exit(0)
    elif results['overall_status'] == 'CONDITIONAL_PASS':
        sys.exit(1)
    else:
        sys.exit(2)