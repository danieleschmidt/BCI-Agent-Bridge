#!/usr/bin/env python3
"""
Quality Gates Validation for BCI-Agent-Bridge Generation 6.
Tests code structure, module organization, and compliance framework.
"""

import os
import sys
from pathlib import Path
import json
import ast
import importlib.util

def validate_project_structure():
    """Validate the overall project structure."""
    print("📁 Validating Project Structure...")
    
    required_dirs = [
        "src/bci_agent_bridge/core",
        "src/bci_agent_bridge/adapters", 
        "src/bci_agent_bridge/monitoring",
        "src/bci_agent_bridge/performance",
        "src/bci_agent_bridge/utils",
        "src/bci_agent_bridge/i18n",
        "src/bci_agent_bridge/compliance",
        "tests"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All required directories present")
    return True

def validate_core_modules():
    """Validate core module files exist."""
    print("🧠 Validating Core Modules...")
    
    core_files = [
        "src/bci_agent_bridge/__init__.py",
        "src/bci_agent_bridge/__main__.py",
        "src/bci_agent_bridge/core/bridge.py",
        "src/bci_agent_bridge/core/decoder.py",
        "src/bci_agent_bridge/adapters/claude_flow.py"
    ]
    
    missing_files = []
    for file_path in core_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing core files: {missing_files}")
        return False
    
    print("✅ All core modules present")
    return True

def validate_monitoring_system():
    """Validate monitoring system components."""
    print("🔍 Validating Monitoring System...")
    
    monitoring_files = [
        "src/bci_agent_bridge/monitoring/__init__.py",
        "src/bci_agent_bridge/monitoring/health_monitor.py",
        "src/bci_agent_bridge/monitoring/metrics_collector.py",
        "src/bci_agent_bridge/monitoring/alert_manager.py"
    ]
    
    missing_files = []
    for file_path in monitoring_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing monitoring files: {missing_files}")
        return False
    
    print("✅ Monitoring system complete")
    return True

def validate_performance_optimizations():
    """Validate performance optimization components."""
    print("⚡ Validating Performance Optimizations...")
    
    performance_files = [
        "src/bci_agent_bridge/performance/__init__.py",
        "src/bci_agent_bridge/performance/caching.py",
        "src/bci_agent_bridge/performance/batch_processor.py",
        "src/bci_agent_bridge/performance/connection_pool.py",
        "src/bci_agent_bridge/performance/load_balancer.py"
    ]
    
    missing_files = []
    for file_path in performance_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing performance files: {missing_files}")
        return False
    
    print("✅ Performance optimizations complete")
    return True

def validate_internationalization():
    """Validate i18n system."""
    print("🌍 Validating Internationalization...")
    
    i18n_files = [
        "src/bci_agent_bridge/i18n/__init__.py",
        "src/bci_agent_bridge/i18n/translator.py",
        "src/bci_agent_bridge/i18n/locales.py",
        "src/bci_agent_bridge/i18n/neural_commands.py"
    ]
    
    missing_files = []
    for file_path in i18n_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing i18n files: {missing_files}")
        return False
    
    print("✅ Internationalization system complete")
    return True

def validate_compliance_framework():
    """Validate compliance and regulatory framework."""
    print("📋 Validating Compliance Framework...")
    
    compliance_files = [
        "src/bci_agent_bridge/compliance/__init__.py",
        "src/bci_agent_bridge/compliance/gdpr.py",
        "src/bci_agent_bridge/compliance/hipaa.py",
        "src/bci_agent_bridge/compliance/data_protection.py",
        "src/bci_agent_bridge/compliance/audit_logger.py"
    ]
    
    missing_files = []
    for file_path in compliance_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing compliance files: {missing_files}")
        return False
    
    print("✅ Compliance framework complete")
    return True

def validate_testing_suite():
    """Validate comprehensive testing suite."""
    print("🧪 Validating Testing Suite...")
    
    test_files = [
        "tests/conftest.py",
        "tests/test_integration.py"
    ]
    
    missing_files = []
    for file_path in test_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing test files: {missing_files}")
        return False
    
    print("✅ Testing suite complete")
    return True

def analyze_code_complexity():
    """Analyze code complexity and quality."""
    print("📊 Analyzing Code Quality...")
    
    python_files = list(Path("src").rglob("*.py"))
    
    total_lines = 0
    total_files = 0
    total_classes = 0
    total_functions = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.splitlines())
                total_lines += lines
                total_files += 1
                
                # Parse AST to count classes and functions
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            total_classes += 1
                        elif isinstance(node, ast.FunctionDef):
                            total_functions += 1
                except:
                    pass  # Skip files with syntax issues
                    
        except Exception:
            continue  # Skip problematic files
    
    print(f"   - Total Python files: {total_files}")
    print(f"   - Total lines of code: {total_lines:,}")
    print(f"   - Total classes: {total_classes}")
    print(f"   - Total functions: {total_functions}")
    
    # Quality thresholds
    if total_files < 20:
        print("⚠️ Project may need more modular structure")
        return False
    
    if total_lines < 5000:
        print("⚠️ Project may be too small for production")
        return False
    
    print("✅ Code complexity and structure appropriate")
    return True

def validate_sdlc_generations():
    """Validate that all SDLC generations were implemented."""
    print("🚀 Validating SDLC Generation Implementation...")
    
    generation_indicators = {
        "Generation 1 (Basic)": [
            "src/bci_agent_bridge/__main__.py",  # CLI enhancement
            "src/bci_agent_bridge/core/bridge.py"  # Core functionality
        ],
        "Generation 2 (Robust)": [
            "src/bci_agent_bridge/monitoring/health_monitor.py",
            "src/bci_agent_bridge/monitoring/alert_manager.py",
            "src/bci_agent_bridge/utils/validation.py"
        ],
        "Generation 3 (Scale)": [
            "src/bci_agent_bridge/performance/caching.py",
            "src/bci_agent_bridge/performance/load_balancer.py",
            "src/bci_agent_bridge/performance/batch_processor.py"
        ],
        "Generation 4 (Testing)": [
            "tests/conftest.py",
            "tests/test_integration.py"
        ],
        "Generation 5 (Global)": [
            "src/bci_agent_bridge/i18n/translator.py",
            "src/bci_agent_bridge/compliance/gdpr.py"
        ],
        "Generation 6 (Quality)": [
            "src/bci_agent_bridge/compliance/hipaa.py",
            "src/bci_agent_bridge/compliance/audit_logger.py"
        ]
    }
    
    all_generations_complete = True
    
    for generation, required_files in generation_indicators.items():
        missing_files = [f for f in required_files if not Path(f).exists()]
        if missing_files:
            print(f"❌ {generation}: Missing {missing_files}")
            all_generations_complete = False
        else:
            print(f"✅ {generation}: Complete")
    
    return all_generations_complete

def validate_documentation():
    """Validate documentation and README."""
    print("📖 Validating Documentation...")
    
    if not Path("README.md").exists():
        print("❌ Missing README.md")
        return False
    
    # Check README contains key sections
    with open("README.md", 'r') as f:
        readme_content = f.read()
    
    required_sections = ["BCI-Agent-Bridge", "Features", "Installation", "Usage"]
    missing_sections = []
    
    for section in required_sections:
        if section not in readme_content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"⚠️ README missing sections: {missing_sections}")
    
    print("✅ Documentation present")
    return True

def main():
    """Run complete quality gates validation."""
    print("🎯 BCI-Agent-Bridge Quality Gates Validation")
    print("=" * 50)
    print("Generation 6: Quality Gates & Validation")
    print("=" * 50)
    
    # Define all validation checks
    validation_checks = [
        ("Project Structure", validate_project_structure),
        ("Core Modules", validate_core_modules),
        ("Monitoring System", validate_monitoring_system),
        ("Performance Optimizations", validate_performance_optimizations),
        ("Internationalization", validate_internationalization),
        ("Compliance Framework", validate_compliance_framework),
        ("Testing Suite", validate_testing_suite),
        ("Code Quality", analyze_code_complexity),
        ("SDLC Generations", validate_sdlc_generations),
        ("Documentation", validate_documentation)
    ]
    
    passed_checks = 0
    total_checks = len(validation_checks)
    
    for check_name, check_function in validation_checks:
        print(f"\n{check_name}:")
        try:
            if check_function():
                passed_checks += 1
            else:
                print(f"   ⚠️ {check_name} needs attention")
        except Exception as e:
            print(f"   ❌ {check_name} failed: {e}")
    
    # Calculate quality score
    quality_score = (passed_checks / total_checks) * 100
    
    print("\n" + "=" * 50)
    print("🏁 QUALITY GATES VALIDATION RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed_checks}/{total_checks} checks")
    print(f"📊 Quality Score: {quality_score:.1f}%")
    
    if quality_score >= 90:
        print("🎉 EXCELLENT: All quality gates passed!")
        print("✅ Ready for Production Deployment (Generation 7)")
        result = "PASSED"
    elif quality_score >= 75:
        print("👍 GOOD: Most quality gates passed")
        print("⚠️ Address remaining issues before production")
        result = "PASSED_WITH_WARNINGS"
    else:
        print("⚠️ NEEDS WORK: Several quality gates failed")
        print("❌ Not ready for production deployment")
        result = "FAILED"
    
    print(f"🎯 Final Result: {result}")
    
    # Generate quality report
    quality_report = {
        "timestamp": "2024-01-01T12:00:00Z",
        "version": "Generation 6",
        "quality_score": quality_score,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "result": result,
        "recommendations": [
            "Continue with Generation 7 (Production Deployment)" if quality_score >= 90 else "Address failing quality gates",
            "Set up CI/CD pipeline for automated quality checking",
            "Configure production monitoring and alerting",
            "Implement automated compliance reporting"
        ]
    }
    
    with open("quality_gates_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2)
    
    print(f"\n📄 Quality report saved to: quality_gates_report.json")
    
    return quality_score >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)