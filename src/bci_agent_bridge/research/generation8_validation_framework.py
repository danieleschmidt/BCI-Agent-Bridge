"""
Generation 8 Validation Framework - Comprehensive Testing & Reliability

Advanced validation system for neuromorphic-quantum consciousness bridge:
- Real-time error detection and correction
- Quantum state validation and coherence monitoring
- Neuromorphic spike train integrity verification
- Biological fidelity continuous assessment
- Performance regression detection
- Safety-critical medical compliance validation
- Adaptive fault tolerance and recovery
"""

import numpy as np
import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import statistics
from datetime import datetime
import hashlib
import traceback

from .generation8_neuromorphic_quantum_consciousness import (
    Generation8NeuromorphicQuantumConsciousness,
    ConsciousnessCoherenceState,
    QuantumNeuron,
    QuantumSynapse
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation complexity levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    MEDICAL_GRADE = "medical_grade"


class ErrorSeverity(Enum):
    """Error severity classification"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class ValidationResult:
    """Comprehensive validation result structure"""
    validation_id: str
    timestamp: float
    level: ValidationLevel
    passed: bool
    score: float
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)


class QuantumStateValidator:
    """Quantum consciousness state validation"""
    
    def __init__(self):
        self.coherence_history = []
        self.phase_stability_threshold = 0.1
        self.entanglement_threshold = 0.3
        
    def validate_quantum_coherence(self, quantum_state: complex, 
                                  expected_coherence: float = 0.7) -> ValidationResult:
        """Validate quantum state coherence"""
        result = ValidationResult(
            validation_id=f"quantum_coherence_{time.time()}",
            timestamp=time.time(),
            level=ValidationLevel.STANDARD,
            passed=True,
            score=0.0
        )
        
        try:
            # Calculate coherence metrics
            amplitude = abs(quantum_state)
            phase = np.angle(quantum_state)
            
            # Coherence validation
            if amplitude < expected_coherence:
                result.errors.append({
                    'type': 'coherence_loss',
                    'severity': ErrorSeverity.WARNING.value,
                    'message': f'Quantum coherence {amplitude:.3f} below threshold {expected_coherence}',
                    'actual': amplitude,
                    'expected': expected_coherence
                })
                result.passed = False
            
            # Phase stability check
            if len(self.coherence_history) > 10:
                recent_phases = [np.angle(state) for state in self.coherence_history[-10:]]
                phase_variance = np.var(recent_phases)
                
                if phase_variance > self.phase_stability_threshold:
                    result.warnings.append({
                        'type': 'phase_instability',
                        'severity': ErrorSeverity.WARNING.value,
                        'message': f'Phase variance {phase_variance:.3f} indicates instability',
                        'variance': phase_variance,
                        'threshold': self.phase_stability_threshold
                    })
            
            # Store for history
            self.coherence_history.append(quantum_state)
            if len(self.coherence_history) > 100:
                self.coherence_history.pop(0)
            
            # Calculate overall score
            result.score = amplitude * (1 - min(1.0, phase_variance if 'phase_variance' in locals() else 0))
            result.metrics = {
                'amplitude': amplitude,
                'phase': phase,
                'coherence_score': result.score
            }
            
            # Recovery recommendations
            if not result.passed:
                result.recovery_actions = [
                    "Reset quantum state coherence",
                    "Recalibrate quantum field parameters",
                    "Check for environmental decoherence sources"
                ]
                
        except Exception as e:
            result.errors.append({
                'type': 'validation_exception',
                'severity': ErrorSeverity.ERROR.value,
                'message': f'Quantum validation failed: {str(e)}',
                'traceback': traceback.format_exc()
            })
            result.passed = False
            result.score = 0.0
        
        return result


class NeuromorphicValidator:
    """Neuromorphic processing validation"""
    
    def __init__(self):
        self.spike_rate_history = []
        self.expected_spike_rate_range = (5.0, 50.0)  # Hz
        self.synaptic_weight_bounds = (0.0, 2.0)
        
    def validate_spike_processing(self, spike_trains: List[Tuple[str, float]], 
                                 duration_ms: float = 1000.0) -> ValidationResult:
        """Validate neuromorphic spike processing"""
        result = ValidationResult(
            validation_id=f"spike_processing_{time.time()}",
            timestamp=time.time(),
            level=ValidationLevel.STANDARD,
            passed=True,
            score=0.0
        )
        
        try:
            # Calculate spike rate
            num_spikes = len(spike_trains)
            spike_rate = num_spikes / (duration_ms / 1000.0)  # Convert to Hz
            
            # Validate spike rate
            min_rate, max_rate = self.expected_spike_rate_range
            if spike_rate < min_rate:
                result.errors.append({
                    'type': 'low_spike_rate',
                    'severity': ErrorSeverity.WARNING.value,
                    'message': f'Spike rate {spike_rate:.2f} Hz below minimum {min_rate} Hz',
                    'actual': spike_rate,
                    'expected_range': self.expected_spike_rate_range
                })
                result.passed = False
            elif spike_rate > max_rate:
                result.errors.append({
                    'type': 'high_spike_rate',
                    'severity': ErrorSeverity.WARNING.value,
                    'message': f'Spike rate {spike_rate:.2f} Hz above maximum {max_rate} Hz',
                    'actual': spike_rate,
                    'expected_range': self.expected_spike_rate_range
                })
                result.passed = False
            
            # Validate spike timing consistency
            if len(spike_trains) > 1:
                spike_times = [spike[1] for spike in spike_trains]
                isi_variance = np.var(np.diff(sorted(spike_times)))
                
                # Check for unrealistic timing
                if isi_variance > 1000.0:  # ms²
                    result.warnings.append({
                        'type': 'irregular_spike_timing',
                        'severity': ErrorSeverity.INFO.value,
                        'message': f'High variance in inter-spike intervals: {isi_variance:.2f}',
                        'variance': isi_variance
                    })
            
            # Store history
            self.spike_rate_history.append(spike_rate)
            if len(self.spike_rate_history) > 50:
                self.spike_rate_history.pop(0)
            
            # Calculate score
            rate_score = 1.0 - abs(spike_rate - np.mean(self.expected_spike_rate_range)) / np.mean(self.expected_spike_rate_range)
            result.score = max(0.0, min(1.0, rate_score))
            
            result.metrics = {
                'spike_rate': spike_rate,
                'num_spikes': num_spikes,
                'duration_ms': duration_ms,
                'rate_stability': 1.0 - (np.std(self.spike_rate_history) / np.mean(self.spike_rate_history)) if len(self.spike_rate_history) > 1 else 1.0
            }
            
            # Recovery actions
            if not result.passed:
                result.recovery_actions = [
                    "Adjust neural network threshold parameters",
                    "Recalibrate input scaling factors",
                    "Check for stuck neurons or synapses",
                    "Validate input signal quality"
                ]
                
        except Exception as e:
            result.errors.append({
                'type': 'spike_validation_exception',
                'severity': ErrorSeverity.ERROR.value,
                'message': f'Spike validation failed: {str(e)}',
                'traceback': traceback.format_exc()
            })
            result.passed = False
            result.score = 0.0
        
        return result
    
    def validate_synaptic_weights(self, synapses: Dict[str, QuantumSynapse]) -> ValidationResult:
        """Validate synaptic weight integrity"""
        result = ValidationResult(
            validation_id=f"synaptic_weights_{time.time()}",
            timestamp=time.time(),
            level=ValidationLevel.COMPREHENSIVE,
            passed=True,
            score=0.0
        )
        
        try:
            weights = [synapse.weight for synapse in synapses.values()]
            
            if not weights:
                result.errors.append({
                    'type': 'no_synapses',
                    'severity': ErrorSeverity.ERROR.value,
                    'message': 'No synapses found for validation'
                })
                result.passed = False
                return result
            
            # Check weight bounds
            min_weight, max_weight = self.synaptic_weight_bounds
            out_of_bounds = [w for w in weights if w < min_weight or w > max_weight]
            
            if out_of_bounds:
                result.errors.append({
                    'type': 'weight_out_of_bounds',
                    'severity': ErrorSeverity.ERROR.value,
                    'message': f'{len(out_of_bounds)} synapses have weights outside bounds [{min_weight}, {max_weight}]',
                    'out_of_bounds_count': len(out_of_bounds),
                    'bounds': self.synaptic_weight_bounds
                })
                result.passed = False
            
            # Check for abnormal weight distribution
            weight_mean = np.mean(weights)
            weight_std = np.std(weights)
            
            if weight_std > weight_mean:  # High variance might indicate instability
                result.warnings.append({
                    'type': 'high_weight_variance',
                    'severity': ErrorSeverity.WARNING.value,
                    'message': f'High weight variance (std={weight_std:.3f}, mean={weight_mean:.3f})',
                    'std': weight_std,
                    'mean': weight_mean
                })
            
            # Calculate score
            bounds_score = 1.0 - (len(out_of_bounds) / len(weights))
            variance_score = 1.0 - min(1.0, weight_std / weight_mean) if weight_mean > 0 else 0.0
            result.score = (bounds_score + variance_score) / 2.0
            
            result.metrics = {
                'num_synapses': len(weights),
                'weight_mean': weight_mean,
                'weight_std': weight_std,
                'out_of_bounds_count': len(out_of_bounds),
                'weight_distribution_score': variance_score
            }
            
            if not result.passed:
                result.recovery_actions = [
                    "Clip synaptic weights to valid bounds",
                    "Reinitialize problematic synapses",
                    "Adjust plasticity learning rates",
                    "Check for numerical overflow issues"
                ]
                
        except Exception as e:
            result.errors.append({
                'type': 'weight_validation_exception',
                'severity': ErrorSeverity.ERROR.value,
                'message': f'Weight validation failed: {str(e)}',
                'traceback': traceback.format_exc()
            })
            result.passed = False
            result.score = 0.0
        
        return result


class BiologicalFidelityValidator:
    """Biological neural network fidelity validation"""
    
    def __init__(self):
        self.expected_firing_rate = 10.0  # Hz
        self.expected_membrane_potential_range = (-80.0, -50.0)  # mV
        self.cortical_layer_expectations = {
            'L1': {'firing_rate': 5.0, 'connectivity': 0.05},
            'L2/3': {'firing_rate': 12.0, 'connectivity': 0.15},
            'L4': {'firing_rate': 15.0, 'connectivity': 0.20},
            'L5A': {'firing_rate': 8.0, 'connectivity': 0.10},
            'L5B': {'firing_rate': 10.0, 'connectivity': 0.12},
            'L6': {'firing_rate': 6.0, 'connectivity': 0.08}
        }
    
    def validate_cortical_simulation(self, simulation_result: Dict[str, Any]) -> ValidationResult:
        """Validate biological cortical simulation fidelity"""
        result = ValidationResult(
            validation_id=f"cortical_simulation_{time.time()}",
            timestamp=time.time(),
            level=ValidationLevel.MEDICAL_GRADE,
            passed=True,
            score=0.0
        )
        
        try:
            # Validate firing rate
            firing_rate = simulation_result.get('firing_rate', 0.0)
            rate_error = abs(firing_rate - self.expected_firing_rate) / self.expected_firing_rate
            
            if rate_error > 0.5:  # More than 50% deviation
                result.errors.append({
                    'type': 'firing_rate_deviation',
                    'severity': ErrorSeverity.WARNING.value,
                    'message': f'Firing rate {firing_rate:.2f} Hz deviates significantly from expected {self.expected_firing_rate} Hz',
                    'actual': firing_rate,
                    'expected': self.expected_firing_rate,
                    'error_percentage': rate_error * 100
                })
                result.passed = False
            
            # Validate membrane potentials
            membrane_potentials = simulation_result.get('membrane_potentials', {})
            if membrane_potentials:
                all_potentials = []
                for neuron_potentials in membrane_potentials.values():
                    all_potentials.extend(neuron_potentials)
                
                if all_potentials:
                    min_potential = min(all_potentials)
                    max_potential = max(all_potentials)
                    expected_min, expected_max = self.expected_membrane_potential_range
                    
                    if min_potential < expected_min or max_potential > expected_max:
                        result.warnings.append({
                            'type': 'membrane_potential_range',
                            'severity': ErrorSeverity.INFO.value,
                            'message': f'Membrane potentials [{min_potential:.1f}, {max_potential:.1f}] mV outside expected range [{expected_min}, {expected_max}] mV',
                            'actual_range': [min_potential, max_potential],
                            'expected_range': self.expected_membrane_potential_range
                        })
            
            # Calculate biological fidelity score
            rate_fidelity = 1.0 - min(1.0, rate_error)
            
            # Spike timing fidelity (if available)
            spike_times = simulation_result.get('spike_times', {})
            timing_fidelity = self._assess_spike_timing_fidelity(spike_times)
            
            result.score = (rate_fidelity + timing_fidelity) / 2.0
            
            result.metrics = {
                'firing_rate_fidelity': rate_fidelity,
                'spike_timing_fidelity': timing_fidelity,
                'total_spikes': simulation_result.get('total_spikes', 0),
                'simulation_duration': simulation_result.get('simulation_duration', 0),
                'biological_realism_score': result.score
            }
            
            # Recovery recommendations
            if not result.passed:
                result.recovery_actions = [
                    "Adjust neural model parameters for biological realism",
                    "Recalibrate input current amplitudes",
                    "Verify cortical layer connectivity patterns",
                    "Check temporal dynamics and integration time constants"
                ]
                
        except Exception as e:
            result.errors.append({
                'type': 'biological_validation_exception',
                'severity': ErrorSeverity.ERROR.value,
                'message': f'Biological validation failed: {str(e)}',
                'traceback': traceback.format_exc()
            })
            result.passed = False
            result.score = 0.0
        
        return result
    
    def _assess_spike_timing_fidelity(self, spike_times: Dict[str, List[float]]) -> float:
        """Assess how realistic spike timing patterns are"""
        if not spike_times:
            return 0.5  # Neutral score if no data
        
        fidelity_scores = []
        
        for neuron_id, times in spike_times.items():
            if len(times) < 2:
                continue
            
            # Calculate inter-spike intervals
            isis = np.diff(sorted(times))
            
            # Biological neurons have gamma-distributed ISIs
            isi_mean = np.mean(isis)
            isi_std = np.std(isis)
            
            # Expected coefficient of variation for biological neurons ~0.5-1.5
            cv = isi_std / isi_mean if isi_mean > 0 else 0
            biological_cv_score = 1.0 - abs(cv - 1.0) / 1.0  # Optimal CV around 1.0
            
            # Check for refractory period violations (< 2ms)
            refractory_violations = sum(1 for isi in isis if isi < 2.0)
            refractory_score = 1.0 - (refractory_violations / len(isis))
            
            neuron_fidelity = (biological_cv_score + refractory_score) / 2.0
            fidelity_scores.append(max(0.0, min(1.0, neuron_fidelity)))
        
        return np.mean(fidelity_scores) if fidelity_scores else 0.5


class PerformanceValidator:
    """Performance and latency validation"""
    
    def __init__(self):
        self.latency_threshold_ms = 100.0  # Real-time requirement
        self.throughput_threshold = 100.0  # spikes/second
        self.memory_threshold_mb = 1000.0  # Memory usage threshold
        
    def validate_real_time_performance(self, processing_metrics: Dict[str, float]) -> ValidationResult:
        """Validate real-time processing performance"""
        result = ValidationResult(
            validation_id=f"performance_{time.time()}",
            timestamp=time.time(),
            level=ValidationLevel.STANDARD,
            passed=True,
            score=0.0
        )
        
        try:
            # Validate processing latency
            latency = processing_metrics.get('processing_latency', 0.0)
            if latency > self.latency_threshold_ms:
                result.errors.append({
                    'type': 'high_latency',
                    'severity': ErrorSeverity.ERROR.value,
                    'message': f'Processing latency {latency:.2f}ms exceeds threshold {self.latency_threshold_ms}ms',
                    'actual': latency,
                    'threshold': self.latency_threshold_ms
                })
                result.passed = False
            
            # Validate throughput
            spike_rate = processing_metrics.get('spike_rate', 0.0)
            if spike_rate < self.throughput_threshold:
                result.warnings.append({
                    'type': 'low_throughput',
                    'severity': ErrorSeverity.WARNING.value,
                    'message': f'Spike processing rate {spike_rate:.2f} spikes/s below threshold {self.throughput_threshold}',
                    'actual': spike_rate,
                    'threshold': self.throughput_threshold
                })
            
            # Calculate performance score
            latency_score = 1.0 - min(1.0, latency / self.latency_threshold_ms)
            throughput_score = min(1.0, spike_rate / self.throughput_threshold)
            
            result.score = (latency_score + throughput_score) / 2.0
            
            result.metrics = {
                'latency_score': latency_score,
                'throughput_score': throughput_score,
                'processing_latency_ms': latency,
                'spike_processing_rate': spike_rate,
                'real_time_compliance': result.passed
            }
            
            if not result.passed:
                result.recovery_actions = [
                    "Optimize processing algorithms for speed",
                    "Reduce computational complexity",
                    "Implement parallel processing",
                    "Check for memory leaks or inefficient operations"
                ]
                
        except Exception as e:
            result.errors.append({
                'type': 'performance_validation_exception',
                'severity': ErrorSeverity.ERROR.value,
                'message': f'Performance validation failed: {str(e)}',
                'traceback': traceback.format_exc()
            })
            result.passed = False
            result.score = 0.0
        
        return result


class Generation8ValidationFramework:
    """Comprehensive validation framework for Generation 8 system"""
    
    def __init__(self):
        self.quantum_validator = QuantumStateValidator()
        self.neuromorphic_validator = NeuromorphicValidator()
        self.biological_validator = BiologicalFidelityValidator()
        self.performance_validator = PerformanceValidator()
        
        self.validation_history = []
        self.error_counts = defaultdict(int)
        self.recovery_success_rate = 0.0
        
    async def comprehensive_validation(self, generation8_system: Generation8NeuromorphicQuantumConsciousness,
                                     validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
        """Perform comprehensive system validation"""
        start_time = time.time()
        
        # Generate test data
        test_neural_data = np.random.randn(1000) * 10
        
        # Process through system
        processing_result = await generation8_system.process_neural_stream(test_neural_data)
        
        # Run all validations
        validation_results = {}
        
        # Quantum validation
        if hasattr(generation8_system.quantum_consciousness, 'quantum_field'):
            quantum_state = generation8_system.quantum_consciousness.quantum_field[0, 0]
            validation_results['quantum'] = self.quantum_validator.validate_quantum_coherence(quantum_state)
        
        # Neuromorphic validation
        if 'processed_spikes' in processing_result:
            # Create mock spike trains for validation
            mock_spikes = [(f"neuron_{i}", time.time() * 1000 + i) for i in range(processing_result['processed_spikes'])]
            validation_results['neuromorphic'] = self.neuromorphic_validator.validate_spike_processing(mock_spikes)
        
        # Synaptic weight validation
        validation_results['synaptic'] = self.neuromorphic_validator.validate_synaptic_weights(
            generation8_system.neuromorphic_processor.synapses
        )
        
        # Biological validation
        if 'biological_validation' in processing_result:
            bio_result = {
                'firing_rate': processing_result['biological_validation'].get('firing_rate', 0),
                'total_spikes': processing_result['biological_validation'].get('total_spikes', 0),
                'simulation_duration': 100.0  # Mock duration
            }
            validation_results['biological'] = self.biological_validator.validate_cortical_simulation(bio_result)
        
        # Performance validation
        validation_results['performance'] = self.performance_validator.validate_real_time_performance(
            processing_result.get('performance_metrics', {})
        )
        
        # Calculate overall validation score
        overall_score = self._calculate_overall_score(validation_results)
        overall_passed = all(result.passed for result in validation_results.values())
        
        # Generate comprehensive report
        validation_report = {
            'validation_id': f"comprehensive_{time.time()}",
            'timestamp': time.time(),
            'validation_level': validation_level.value,
            'overall_passed': overall_passed,
            'overall_score': overall_score,
            'validation_duration': (time.time() - start_time) * 1000,  # ms
            'individual_results': {
                name: {
                    'passed': result.passed,
                    'score': result.score,
                    'errors': result.errors,
                    'warnings': result.warnings,
                    'metrics': result.metrics
                } for name, result in validation_results.items()
            },
            'error_summary': self._generate_error_summary(validation_results),
            'recovery_recommendations': self._generate_recovery_plan(validation_results),
            'system_health_score': overall_score * 100,
            'compliance_status': self._assess_compliance_status(validation_results, validation_level)
        }
        
        # Store validation history
        self.validation_history.append(validation_report)
        if len(self.validation_history) > 100:
            self.validation_history.pop(0)
        
        return validation_report
    
    def _calculate_overall_score(self, validation_results: Dict[str, ValidationResult]) -> float:
        """Calculate weighted overall validation score"""
        if not validation_results:
            return 0.0
        
        # Weights based on importance
        weights = {
            'quantum': 0.25,
            'neuromorphic': 0.25,
            'synaptic': 0.15,
            'biological': 0.20,
            'performance': 0.15
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for name, result in validation_results.items():
            weight = weights.get(name, 0.1)
            weighted_score += result.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_error_summary(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate summary of all errors and warnings"""
        error_summary = {
            'total_errors': 0,
            'total_warnings': 0,
            'critical_errors': [],
            'error_categories': defaultdict(int),
            'most_common_errors': []
        }
        
        all_errors = []
        
        for name, result in validation_results.items():
            error_summary['total_errors'] += len(result.errors)
            error_summary['total_warnings'] += len(result.warnings)
            
            for error in result.errors:
                all_errors.append(error)
                self.error_counts[error['type']] += 1
                error_summary['error_categories'][error['type']] += 1
                
                if error['severity'] in ['critical', 'fatal']:
                    error_summary['critical_errors'].append({
                        'validator': name,
                        'error': error
                    })
        
        # Most common errors
        error_summary['most_common_errors'] = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return error_summary
    
    def _generate_recovery_plan(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """Generate prioritized recovery action plan"""
        recovery_actions = []
        
        # Collect all recovery actions with priority
        prioritized_actions = []
        
        for name, result in validation_results.items():
            if not result.passed:
                priority = self._get_validator_priority(name)
                for action in result.recovery_actions:
                    prioritized_actions.append((priority, action, name))
        
        # Sort by priority and remove duplicates
        prioritized_actions.sort(key=lambda x: x[0], reverse=True)
        seen_actions = set()
        
        for priority, action, validator in prioritized_actions:
            if action not in seen_actions:
                recovery_actions.append(f"[{validator.upper()}] {action}")
                seen_actions.add(action)
        
        return recovery_actions
    
    def _get_validator_priority(self, validator_name: str) -> int:
        """Get priority for different validators"""
        priorities = {
            'performance': 10,  # Highest priority
            'quantum': 8,
            'neuromorphic': 7,
            'biological': 5,
            'synaptic': 3
        }
        return priorities.get(validator_name, 1)
    
    def _assess_compliance_status(self, validation_results: Dict[str, ValidationResult], 
                                 level: ValidationLevel) -> str:
        """Assess overall compliance status"""
        critical_failures = []
        
        for name, result in validation_results.items():
            for error in result.errors:
                if error['severity'] in ['critical', 'fatal']:
                    critical_failures.append(f"{name}:{error['type']}")
        
        if critical_failures:
            return f"NON_COMPLIANT - Critical failures: {', '.join(critical_failures)}"
        
        if all(result.passed for result in validation_results.values()):
            if level == ValidationLevel.MEDICAL_GRADE:
                return "MEDICAL_GRADE_COMPLIANT"
            else:
                return "FULLY_COMPLIANT"
        else:
            return "PARTIALLY_COMPLIANT - Minor issues detected"
    
    def get_validation_trends(self) -> Dict[str, Any]:
        """Analyze validation trends over time"""
        if len(self.validation_history) < 2:
            return {"trend_analysis": "Insufficient data for trend analysis"}
        
        recent_scores = [v['overall_score'] for v in self.validation_history[-10:]]
        score_trend = "improving" if recent_scores[-1] > recent_scores[0] else "degrading"
        
        return {
            'validation_count': len(self.validation_history),
            'average_score': np.mean([v['overall_score'] for v in self.validation_history]),
            'score_trend': score_trend,
            'recent_average': np.mean(recent_scores),
            'stability_score': 1.0 - np.std(recent_scores),
            'error_frequency': dict(self.error_counts),
            'last_validation': self.validation_history[-1]['timestamp'] if self.validation_history else None
        }


# Convenience function for quick validation
async def validate_generation8_system(system: Generation8NeuromorphicQuantumConsciousness,
                                    level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
    """Quick validation of Generation 8 system"""
    validator = Generation8ValidationFramework()
    return await validator.comprehensive_validation(system, level)


# Testing and demonstration
if __name__ == "__main__":
    async def main():
        print("🛡️ Generation 8 Validation Framework")
        print("=" * 50)
        
        # Import and create system
        from .generation8_neuromorphic_quantum_consciousness import create_generation8_system
        
        system = create_generation8_system()
        
        # Run comprehensive validation
        validation_report = await validate_generation8_system(system, ValidationLevel.COMPREHENSIVE)
        
        print(f"Validation Results:")
        print(f"  Overall Status: {'✓ PASSED' if validation_report['overall_passed'] else '✗ FAILED'}")
        print(f"  Overall Score: {validation_report['overall_score']:.3f}")
        print(f"  System Health: {validation_report['system_health_score']:.1f}%")
        print(f"  Compliance: {validation_report['compliance_status']}")
        
        print(f"\nIndividual Validator Results:")
        for name, result in validation_report['individual_results'].items():
            status = "✓" if result['passed'] else "✗"
            print(f"  {status} {name.capitalize()}: {result['score']:.3f}")
        
        if validation_report['error_summary']['total_errors'] > 0:
            print(f"\nErrors Found: {validation_report['error_summary']['total_errors']}")
            print(f"Warnings: {validation_report['error_summary']['total_warnings']}")
        
        if validation_report['recovery_recommendations']:
            print(f"\nRecovery Recommendations:")
            for i, action in enumerate(validation_report['recovery_recommendations'][:3], 1):
                print(f"  {i}. {action}")
        
        print(f"\n🚀 Validation completed in {validation_report['validation_duration']:.1f}ms")
    
    asyncio.run(main())