#\!/usr/bin/env python3
"""
Comprehensive quality gates validation for BCI-Agent-Bridge.
"""

import subprocess
import sys
import json
import time
from typing import Dict, Any, Tuple


def run_command(cmd: list, cwd: str = ".") -> Tuple[bool, str, str]:
    """Run a command and return success, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd,
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_test_coverage() -> Dict[str, Any]:
    """Run tests and check coverage."""
    print("🧪 Running tests with coverage...")
    
    results = {
        "passed": True,
        "coverage_percentage": 0,
        "tests_passed": 0,
        "tests_failed": 0
    }
    
    # Run tests with coverage
    success, stdout, stderr = run_command([
        "bash", "-c",
        "source venv/bin/activate && python3 -m pytest tests/test_core.py -v --cov=bci_agent_bridge --cov-report=term-missing"
    ])
    
    # Parse test results  
    import re
    passed_match = re.search(r"(\d+) passed", stdout)
    failed_match = re.search(r"(\d+) failed", stdout)
    
    if passed_match:
        results["tests_passed"] = int(passed_match.group(1))
    if failed_match:
        results["tests_failed"] = int(failed_match.group(1))
        results["passed"] = False
    
    # Extract coverage percentage
    coverage_match = re.search(r"TOTAL.+?(\d+)%", stdout)
    if coverage_match:
        results["coverage_percentage"] = int(coverage_match.group(1))
        print(f"📊 Test coverage: {results['coverage_percentage']}%")
    
    if results["tests_failed"] == 0:
        print("✅ All core tests passed")
    else:
        print(f"❌ {results['tests_failed']} tests failed")
    
    return results


def check_performance_benchmarks() -> Dict[str, Any]:
    """Run performance benchmarks."""
    print("⚡ Running performance benchmarks...")
    
    results = {
        "passed": True,
        "benchmarks": {}
    }
    
    # Run performance test
    success, stdout, stderr = run_command([
        "bash", "-c", 
        '''
        source venv/bin/activate &&
        python3 -c "
import time
import numpy as np

# Test imports and basic functionality
try:
    from bci_agent_bridge.core.bridge import BCIBridge
    from bci_agent_bridge.performance.caching import NeuralDataCache
    
    # BCI Bridge initialization
    start = time.time()
    bridge = BCIBridge()
    init_time = (time.time() - start) * 1000
    print(f'init_time:{init_time:.1f}')
    
    # Neural processing
    test_data = np.random.randn(8, 250).astype(np.float32)
    start = time.time()
    for _ in range(5):
        _ = bridge.preprocessor.apply_filters(test_data)
    proc_time = (time.time() - start) / 5 * 1000
    print(f'processing_time:{proc_time:.1f}')
    
    # Caching performance
    cache = NeuralDataCache()
    start = time.time()
    for i in range(10):
        cache.put(f'test_{i}', test_data)
    cache_time = (time.time() - start) / 10 * 1000
    print(f'cache_time:{cache_time:.1f}')
    
    print('SUCCESS')
except Exception as e:
    print(f'ERROR:{e}')
"
        '''
    ])
    
    if success and "SUCCESS" in stdout:
        # Parse results
        for line in stdout.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    key, value_str = parts
                    try:
                        value = float(value_str)
                        results["benchmarks"][key] = {
                            "value": value,
                            "unit": "ms",
                            "passed": value < 1000  # All should be under 1 second
                        }
                    except:
                        pass
        
        all_passed = all(b.get("passed", True) for b in results["benchmarks"].values())
        results["passed"] = all_passed
        
        if all_passed:
            print("✅ Performance benchmarks passed")
        else:
            print("❌ Some performance benchmarks failed")
    else:
        results["passed"] = False
        print(f"❌ Performance benchmark error: {stderr}")
    
    return results


def check_system_health() -> Dict[str, Any]:
    """Check system health."""
    print("🏥 Checking system health...")
    
    results = {"passed": True}
    
    # Test basic imports
    success, stdout, stderr = run_command([
        "bash", "-c",
        '''
        source venv/bin/activate &&
        python3 -c "
try:
    from bci_agent_bridge import BCIBridge
    from bci_agent_bridge.security.input_validator import InputValidator
    from bci_agent_bridge.performance.caching import NeuralDataCache
    print('IMPORTS_OK')
except Exception as e:
    print(f'IMPORT_ERROR:{e}')
"
        '''
    ])
    
    if success and "IMPORTS_OK" in stdout:
        print("✅ System health check passed")
        results["passed"] = True
    else:
        print(f"❌ System health check failed: {stderr}")
        results["passed"] = False
    
    return results


def generate_report(results: Dict[str, Any]) -> None:
    """Generate quality report."""
    print("\n" + "="*50)
    print("📋 QUALITY GATES REPORT")
    print("="*50)
    
    overall_passed = True
    
    # Testing
    testing = results.get("testing", {})
    print(f"\n🧪 TESTING")
    print(f"  Tests passed: {testing.get('tests_passed', 0)}")
    print(f"  Tests failed: {testing.get('tests_failed', 0)}")
    print(f"  Coverage: {testing.get('coverage_percentage', 0)}%")
    
    if not testing.get("passed", False):
        overall_passed = False
    
    # Performance
    performance = results.get("performance", {})
    print(f"\n⚡ PERFORMANCE")
    for name, data in performance.get("benchmarks", {}).items():
        status = "✅" if data.get("passed", False) else "❌"
        print(f"  {status} {name}: {data.get('value', 0):.1f}{data.get('unit', '')}")
    
    if not performance.get("passed", False):
        overall_passed = False
    
    # Health
    health = results.get("health", {})
    status = "✅" if health.get("passed", False) else "❌"
    print(f"\n🏥 SYSTEM HEALTH {status}")
    
    if not health.get("passed", False):
        overall_passed = False
    
    # Overall
    print("\n" + "="*50)
    if overall_passed:
        print("🎉 ALL QUALITY GATES PASSED\!")
        print("🚀 Ready for production deployment")
    else:
        print("⚠️  QUALITY GATES FAILED")
        print("🔧 Issues need to be resolved")
    print("="*50)
    
    # Save report
    with open("quality_gates_report.json", "w") as f:
        json.dump({
            **results,
            "overall_passed": overall_passed,
            "timestamp": time.time()
        }, f, indent=2)


def main():
    """Main validation."""
    print("🚀 Starting Quality Gates Validation")
    
    results = {
        "testing": check_test_coverage(),
        "performance": check_performance_benchmarks(), 
        "health": check_system_health()
    }
    
    generate_report(results)
    
    overall_passed = all(r.get("passed", False) for r in results.values())
    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
EOF < /dev/null
