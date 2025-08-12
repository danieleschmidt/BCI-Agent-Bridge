#!/usr/bin/env python3
"""
Comprehensive security vulnerability scan for BCI-Agent-Bridge
"""

import json
import time
import numpy as np

def run_security_scan():
    print('🔒 Running comprehensive security scan...')

    security_results = {
        'timestamp': time.time(),
        'scans_completed': [],
        'vulnerabilities': [],
        'security_score': 0
    }

    # Test 1: Input validation security
    print('🔍 Testing input validation...')
    try:
        from bci_agent_bridge.security.input_validator import InputValidator, SecurityPolicy
        validator = InputValidator(SecurityPolicy.CLINICAL)
        
        # Test malicious inputs
        malicious_inputs = [
            '<script>alert("xss")</script>',
            'DROP TABLE users;',
            '../../../etc/passwd',
            'A' * 10000  # Buffer overflow attempt
        ]
        
        validation_passed = 0
        for malicious_input in malicious_inputs:
            try:
                result = validator.validate_string_input(malicious_input, 'test_field')
                if result != malicious_input:  # Input was sanitized
                    validation_passed += 1
            except Exception:
                validation_passed += 1  # Exception means input was rejected
        
        security_results['scans_completed'].append('input_validation')
        if validation_passed == len(malicious_inputs):
            print('✅ Input validation: ALL MALICIOUS INPUTS BLOCKED')
            security_results['security_score'] += 20
        else:
            print(f'⚠️  Input validation: {validation_passed}/{len(malicious_inputs)} blocked')
            security_results['vulnerabilities'].append('weak_input_validation')
            
    except Exception as e:
        print(f'❌ Input validation test failed: {e}')
        security_results['vulnerabilities'].append('input_validation_error')

    # Test 2: Data privacy protection
    print('🔍 Testing data privacy protection...')
    try:
        from bci_agent_bridge.privacy.differential_privacy import DifferentialPrivacy
        
        privacy_engine = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        sensitive_data = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        private_data = privacy_engine.add_noise(sensitive_data, sensitivity=1.0)
        
        # Check that noise was added (data should be different)
        if not np.array_equal(sensitive_data, private_data):
            print('✅ Differential privacy: NOISE SUCCESSFULLY ADDED')
            security_results['security_score'] += 25
            security_results['scans_completed'].append('differential_privacy')
        else:
            print('⚠️  Differential privacy: No noise detected')
            security_results['vulnerabilities'].append('weak_privacy_protection')
            
    except Exception as e:
        print(f'❌ Privacy protection test failed: {e}')
        security_results['vulnerabilities'].append('privacy_protection_error')

    # Test 3: Secure neural data handling
    print('🔍 Testing secure neural data handling...')
    try:
        from bci_agent_bridge import BCIBridge
        from bci_agent_bridge.core.bridge import NeuralData
        
        # Test with privacy mode enabled
        bridge = BCIBridge(privacy_mode=True)
        test_data = np.random.randn(8, 250)
        
        neural_data = NeuralData(
            data=test_data,
            timestamp=time.time(),
            channels=[f'CH{i}' for i in range(1, 9)],
            sampling_rate=250
        )
        
        intention = bridge.decode_intention(neural_data)
        
        # Check that neural features are not exposed in privacy mode
        if intention.neural_features is None:
            print('✅ Neural privacy: FEATURES PROTECTED IN PRIVACY MODE')
            security_results['security_score'] += 20
            security_results['scans_completed'].append('neural_privacy')
        else:
            print('⚠️  Neural privacy: Features exposed in privacy mode')
            security_results['vulnerabilities'].append('neural_data_exposure')
            
    except Exception as e:
        print(f'❌ Neural privacy test failed: {e}')
        security_results['vulnerabilities'].append('neural_privacy_error')

    # Test 4: Security logging
    print('🔍 Testing security audit logging...')
    try:
        from bci_agent_bridge.security.audit_logger import security_logger
        
        # Test security event logging
        security_logger.log_suspicious_activity(
            activity_type='test_security_scan',
            details={'scan_type': 'automated'},
            risk_score=1
        )
        
        print('✅ Security logging: AUDIT TRAIL ACTIVE')
        security_results['security_score'] += 20
        security_results['scans_completed'].append('security_logging')
        
    except Exception as e:
        print(f'❌ Security logging test failed: {e}')
        security_results['vulnerabilities'].append('security_logging_error')

    # Test 5: Error handling and information disclosure
    print('🔍 Testing error handling security...')
    try:
        from bci_agent_bridge import BCIBridge
        
        # Test with invalid parameters to trigger errors
        try:
            bridge = BCIBridge(channels=-1)  # Invalid parameter
            print('⚠️  Error handling: Invalid parameters accepted')
            security_results['vulnerabilities'].append('weak_parameter_validation')
        except ValueError as e:
            error_msg = str(e)
            # Check that error doesn't expose sensitive system information
            if 'Invalid BCIBridge configuration' in error_msg:
                print('✅ Error handling: SECURE ERROR MESSAGES')
                security_results['security_score'] += 15
                security_results['scans_completed'].append('error_handling')
            else:
                print('⚠️  Error handling: Potentially sensitive error details')
                security_results['vulnerabilities'].append('information_disclosure')
                
    except Exception as e:
        print(f'❌ Error handling test failed: {e}')
        security_results['vulnerabilities'].append('error_handling_error')

    # Final security assessment
    print('')
    print('🔒 SECURITY SCAN RESULTS:')
    print('=' * 50)
    print(f'Security Score: {security_results["security_score"]}/100')
    print(f'Scans Completed: {len(security_results["scans_completed"])}')
    print(f'Vulnerabilities Found: {len(security_results["vulnerabilities"])}')

    if security_results['security_score'] >= 85:
        print('✅ SECURITY STATUS: EXCELLENT')
    elif security_results['security_score'] >= 70:
        print('⚠️  SECURITY STATUS: GOOD (minor issues)')
    else:
        print('❌ SECURITY STATUS: NEEDS IMPROVEMENT')

    if security_results['vulnerabilities']:
        print('')
        print('⚠️  VULNERABILITIES DETECTED:')
        for vuln in security_results['vulnerabilities']:
            print(f'  - {vuln}')

    # Save security report
    with open('security_report.json', 'w') as f:
        json.dump(security_results, f, indent=2)
    print('')
    print('📄 Security report saved to security_report.json')
    
    return security_results

if __name__ == '__main__':
    run_security_scan()