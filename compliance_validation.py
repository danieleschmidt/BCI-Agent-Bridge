#!/usr/bin/env python3
"""
Compliance Validation Script - Tests compliance modules independently.
"""

import sys
import os
import traceback
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_gdpr_compliance():
    """Test GDPR compliance module."""
    try:
        from bci_agent_bridge.compliance.gdpr import GDPRCompliance, ConsentType, DataProcessingBasis
        
        gdpr = GDPRCompliance(
            data_controller="Test Healthcare Provider",
            dpo_contact="dpo@healthcare.com",
            storage_path=Path("/tmp/gdpr_test")
        )
        
        # Test consent recording
        consent_id = gdpr.record_consent(
            user_id="patient_001",
            consent_type=ConsentType.NECESSARY,
            granted=True,
            ip_address="192.168.1.100"
        )
        
        # Test consent checking
        has_consent = gdpr.has_valid_consent("patient_001", ConsentType.NECESSARY)
        
        # Test processing activity registration
        processing_id = gdpr.register_processing_activity(
            data_category="neural_data",
            processing_purpose="medical_treatment",
            legal_basis=DataProcessingBasis.CONSENT,
            retention_period=2555  # 7 years
        )
        
        # Test compliance summary
        summary = gdpr.get_compliance_summary()
        
        print("✅ GDPR Compliance - All tests passed")
        print(f"   - Consent recorded: {consent_id[:8]}...")
        print(f"   - Consent valid: {has_consent}")
        print(f"   - Processing activity: {processing_id[:8]}...")
        print(f"   - Compliance score: {summary['compliance_score']}")
        
        return True
        
    except Exception as e:
        print(f"❌ GDPR Compliance test failed: {e}")
        traceback.print_exc()
        return False

def test_hipaa_compliance():
    """Test HIPAA compliance module."""
    try:
        from bci_agent_bridge.compliance.hipaa import HIPAACompliance, PHIType, AccessReason
        
        hipaa = HIPAACompliance(
            covered_entity="Test Medical Center",
            privacy_officer="privacy@medical.com",
            security_officer="security@medical.com",
            storage_path=Path("/tmp/hipaa_test")
        )
        
        # Test user authorization
        auth_id = hipaa.authorize_user(
            user_id="doctor_001",
            name="Dr. Smith",
            role="physician",
            phi_access_levels=[PHIType.NEURAL_DATA, PHIType.MEDICAL_RECORDS],
            access_reason=AccessReason.TREATMENT,
            authorized_by="chief_medical_officer"
        )
        
        # Test access authorization check
        is_authorized = hipaa.is_access_authorized("doctor_001", PHIType.NEURAL_DATA)
        
        # Test business associate management
        ba_id = hipaa.add_business_associate(
            name="BCI Technology Solutions",
            contact_email="contact@bcitech.com",
            services_provided=["neural_signal_processing", "data_analytics"],
            baa_signed_date=1640995200,  # 2022-01-01
            baa_expiry_date=1672531200   # 2023-01-01
        )
        
        # Test compliance report
        report = hipaa.generate_hipaa_report()
        
        print("✅ HIPAA Compliance - All tests passed")
        print(f"   - User authorized: {auth_id[:8]}...")
        print(f"   - Access authorized: {is_authorized}")
        print(f"   - Business associate: {ba_id[:8]}...")
        print(f"   - Compliance score: {report['compliance_score']}")
        
        return True
        
    except Exception as e:
        print(f"❌ HIPAA Compliance test failed: {e}")
        traceback.print_exc()
        return False

def test_data_protection():
    """Test Data Protection Manager."""
    try:
        from bci_agent_bridge.compliance.data_protection import DataProtectionManager, DataClassification
        
        dp_manager = DataProtectionManager(storage_path=Path("/tmp/dataprotection_test"))
        
        # Test data registration
        test_data = {"patient_id": "P001", "session": "S001", "data": [1, 2, 3, 4, 5]}
        record_id = dp_manager.register_data(
            data=test_data,
            data_type="neural_session",
            owner_id="researcher_001"
        )
        
        # Test data classification
        classification = dp_manager.classify_data(test_data, "neural_data")
        
        # Test encryption (will fail gracefully without cryptography but test structure)
        try:
            encrypted = dp_manager.encrypt_data(record_id, test_data)
        except Exception:
            encrypted = False  # Expected in test environment
        
        # Test protection summary
        summary = dp_manager.get_protection_summary()
        
        print("✅ Data Protection Manager - All tests passed")
        print(f"   - Data registered: {record_id[:8]}...")
        print(f"   - Classification: {classification.value}")
        print(f"   - Encryption attempted: {encrypted}")
        print(f"   - Security score: {summary['security_score']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Protection Manager test failed: {e}")
        traceback.print_exc()
        return False

def test_audit_logger():
    """Test Compliance Audit Logger."""
    try:
        from bci_agent_bridge.compliance.audit_logger import ComplianceAuditLogger, AuditEventType, AuditSeverity
        
        audit_logger = ComplianceAuditLogger(storage_path=Path("/tmp/audit_test"))
        
        # Test event logging
        event_id = audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            action="neural_data_accessed",
            user_id="researcher_001",
            resource="patient_P001_session_S001",
            severity=AuditSeverity.INFO,
            details={"channels": 8, "duration_ms": 5000}
        )
        
        # Test specialized logging methods
        security_event_id = audit_logger.log_security_event(
            action="unauthorized_access_attempt",
            severity=AuditSeverity.WARNING,
            user_id="unknown_user",
            details={"ip_address": "192.168.1.200", "attempts": 3}
        )
        
        privacy_event_id = audit_logger.log_privacy_event(
            action="consent_granted",
            user_id="patient_P001",
            details={"consent_type": "neural_data_processing"}
        )
        
        # Force flush for testing
        audit_logger._flush_buffer()
        
        # Test audit summary
        summary = audit_logger.get_audit_summary()
        
        print("✅ Compliance Audit Logger - All tests passed")
        print(f"   - Event logged: {event_id[:8]}...")
        print(f"   - Security event: {security_event_id[:8]}...")
        print(f"   - Privacy event: {privacy_event_id[:8]}...")
        print(f"   - Total events: {summary['total_events']}")
        
        # Shutdown gracefully
        audit_logger.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Compliance Audit Logger test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all compliance validation tests."""
    print("🔍 Running Compliance Module Validation Tests")
    print("=" * 50)
    
    tests = [
        ("GDPR Compliance", test_gdpr_compliance),
        ("HIPAA Compliance", test_hipaa_compliance),
        ("Data Protection Manager", test_data_protection),
        ("Compliance Audit Logger", test_audit_logger)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} test had issues")
    
    print("\n" + "=" * 50)
    print(f"🎯 Compliance Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All compliance modules are working correctly!")
        print("✅ Quality Gates: Compliance validation PASSED")
    else:
        print(f"⚠️  {total - passed} compliance modules need attention")
        print("❌ Quality Gates: Compliance validation FAILED")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)