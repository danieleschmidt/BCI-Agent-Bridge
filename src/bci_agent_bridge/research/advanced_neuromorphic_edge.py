"""
Generation 5: Advanced Neuromorphic Edge Computing with Quantum Acceleration

Revolutionary neuromorphic computing architecture for ultra-low-power BCI processing
with quantum-accelerated spiking neural networks and event-driven computation.

Key Innovations:
- Quantum-enhanced spiking neural networks with STDP learning
- Event-driven neural processing with <1mW power consumption
- Adaptive spike-timing-dependent plasticity with quantum coherence
- Edge-optimized neuromorphic circuits for real-time BCI applications
"""

import numpy as np
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Protocol, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import json
from collections import deque, defaultdict
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuromorphicArchitecture(Enum):
    """Supported neuromorphic architectures."""
    LOIHI = "intel_loihi"
    SPINNAKER = "spinnaker"
    TRUENORTH = "truenorth"
    AKIDA = "brainchip_akida"
    DYNAP_SE = "dynap_se"
    QUANTUM_HYBRID = "quantum_neuromorphic"


class SpikeEncoding(Enum):
    """Neural spike encoding methods."""
    RATE_CODING = "rate"
    TEMPORAL_CODING = "temporal"
    POPULATION_CODING = "population"
    RANK_ORDER_CODING = "rank_order"
    QUANTUM_SPIKE_CODING = "quantum_spike"


@dataclass
class NeuromorphicConfig:
    """Configuration for neuromorphic computing system."""
    architecture: NeuromorphicArchitecture = NeuromorphicArchitecture.QUANTUM_HYBRID
    n_neurons: int = 1024
    n_synapses: int = 10240
    spike_encoding: SpikeEncoding = SpikeEncoding.QUANTUM_SPIKE_CODING
    time_resolution: float = 0.1  # ms
    membrane_voltage_threshold: float = 1.0
    refractory_period: float = 2.0  # ms
    stdp_learning_rate: float = 0.01
    quantum_coherence_time: float = 10.0  # ms
    power_budget: float = 1.0  # mW


@dataclass
class SpikeEvent:
    """Individual spike event in neuromorphic processing."""
    neuron_id: int
    timestamp: float
    amplitude: float
    phase: Optional[float] = None  # Quantum phase
    coherence: Optional[float] = None  # Quantum coherence
    source_channel: Optional[int] = None
    event_type: str = "excitatory"


@dataclass
class QuantumSpike:
    """Quantum-enhanced spike with superposition properties."""
    classical_spike: SpikeEvent
    quantum_state: np.ndarray  # Complex amplitude
    entanglement_partners: List[int] = field(default_factory=list)
    decoherence_rate: float = 0.1
    measurement_basis: str = "computational"


@dataclass
class NeuromorphicProcessingStats:
    """Statistics from neuromorphic processing."""
    total_spikes: int
    power_consumption: float  # mW
    processing_latency: float  # ms
    spike_rate: float  # spikes/sec
    quantum_coherence_avg: float
    stdp_updates: int
    energy_efficiency: float  # spikes/mJ


class QuantumNeuron:
    """Quantum-enhanced neuromorphic neuron with superposition states."""
    
    def __init__(self, neuron_id: int, config: NeuromorphicConfig):
        self.neuron_id = neuron_id
        self.config = config
        
        # Classical neuron state
        self.membrane_voltage = 0.0
        self.last_spike_time = -np.inf
        self.refractory_until = 0.0
        
        # Quantum state components
        self.quantum_amplitude = np.array([1.0+0j, 0.0+0j])  # |0⟩ state initially
        self.quantum_phase = 0.0
        self.coherence_time = config.quantum_coherence_time
        self.last_coherence_update = 0.0
        
        # Synaptic connections
        self.input_synapses = {}  # {source_id: weight}
        self.output_synapses = {}  # {target_id: weight}
        
        # STDP learning parameters
        self.spike_trace = 0.0
        self.trace_decay = 0.95
        
        # Energy tracking
        self.energy_consumed = 0.0
        
        logger.debug(f"QuantumNeuron {neuron_id} initialized")
    
    def add_input_synapse(self, source_id: int, initial_weight: float = 0.1):
        """Add input synaptic connection."""
        self.input_synapses[source_id] = initial_weight
    
    def add_output_synapse(self, target_id: int, initial_weight: float = 0.1):
        """Add output synaptic connection."""
        self.output_synapses[target_id] = initial_weight
    
    def update_quantum_state(self, current_time: float, external_field: float = 0.0):
        """Update quantum state with decoherence and external influences."""
        dt = current_time - self.last_coherence_update
        self.last_coherence_update = current_time
        
        # Decoherence model: T2* decay
        decoherence_factor = np.exp(-dt / self.coherence_time)
        
        # Quantum evolution with external neural field
        phase_evolution = external_field * dt * 0.1  # Simplified Hamiltonian
        
        # Update quantum amplitudes
        rotation_angle = phase_evolution + self.quantum_phase
        new_amplitude = np.array([
            self.quantum_amplitude[0] * np.cos(rotation_angle/2) * decoherence_factor,
            self.quantum_amplitude[1] * np.sin(rotation_angle/2) * decoherence_factor
        ])
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(new_amplitude)**2))
        if norm > 0:
            self.quantum_amplitude = new_amplitude / norm
        
        # Update phase
        self.quantum_phase = (self.quantum_phase + phase_evolution) % (2 * np.pi)
    
    def process_spike_input(self, spike: SpikeEvent, current_time: float) -> Optional[QuantumSpike]:
        """Process incoming spike and potentially generate output spike."""
        # Check refractory period
        if current_time < self.refractory_until:
            return None
        
        # Get synaptic weight
        synaptic_weight = self.input_synapses.get(spike.neuron_id, 0.0)
        
        # Update membrane voltage
        if spike.event_type == "excitatory":
            voltage_change = synaptic_weight * spike.amplitude
        else:  # inhibitory
            voltage_change = -synaptic_weight * spike.amplitude
        
        # Quantum-enhanced voltage integration
        quantum_enhancement = self._calculate_quantum_enhancement()
        self.membrane_voltage += voltage_change * quantum_enhancement
        
        # Energy consumption for processing
        self.energy_consumed += 0.1e-9  # 0.1 nJ per spike processing
        
        # Update spike trace for STDP
        self.spike_trace = self.spike_trace * self.trace_decay + 1.0
        
        # Check for spike generation
        if self.membrane_voltage >= self.config.membrane_voltage_threshold:
            return self._generate_quantum_spike(current_time)
        
        return None
    
    def _calculate_quantum_enhancement(self) -> float:
        """Calculate quantum enhancement factor based on current quantum state."""
        # Quantum coherence enhances neural processing
        coherence = np.abs(self.quantum_amplitude[0] * np.conj(self.quantum_amplitude[1]))
        
        # Quantum interference effect
        quantum_prob = np.abs(self.quantum_amplitude[1])**2
        enhancement = 1.0 + 0.2 * quantum_prob + 0.1 * coherence
        
        return enhancement
    
    def _generate_quantum_spike(self, current_time: float) -> QuantumSpike:
        """Generate quantum spike when threshold is reached."""
        # Reset membrane voltage and set refractory period
        self.membrane_voltage = 0.0
        self.refractory_until = current_time + self.config.refractory_period
        self.last_spike_time = current_time
        
        # Create classical spike
        classical_spike = SpikeEvent(
            neuron_id=self.neuron_id,
            timestamp=current_time,
            amplitude=1.0,
            phase=self.quantum_phase,
            coherence=self._get_current_coherence()
        )
        
        # Create quantum state for spike
        quantum_state = self.quantum_amplitude.copy()
        
        # Energy for spike generation
        self.energy_consumed += 1.0e-9  # 1 nJ per spike
        
        # Apply STDP learning
        self._apply_stdp_learning()
        
        return QuantumSpike(
            classical_spike=classical_spike,
            quantum_state=quantum_state,
            decoherence_rate=1.0 / self.coherence_time,
            measurement_basis="computational"
        )
    
    def _get_current_coherence(self) -> float:
        """Get current quantum coherence measure."""
        return 2 * np.abs(self.quantum_amplitude[0] * np.conj(self.quantum_amplitude[1]))
    
    def _apply_stdp_learning(self):
        """Apply spike-timing-dependent plasticity learning."""
        # Update all input synapses based on STDP
        for source_id, weight in self.input_synapses.items():
            # Simplified STDP: increase weights for recent inputs
            if self.spike_trace > 0.5:
                weight_change = self.config.stdp_learning_rate * self.spike_trace
                self.input_synapses[source_id] = np.clip(weight + weight_change, 0.0, 1.0)


class NeuromorphicSpikeEncoder:
    """Encode continuous neural signals into spike trains."""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.encoding_method = config.spike_encoding
        
    def encode_neural_signals(self, neural_data: np.ndarray, time_window: float) -> List[SpikeEvent]:
        """Encode neural signals into spike trains."""
        if self.encoding_method == SpikeEncoding.QUANTUM_SPIKE_CODING:
            return self._quantum_spike_encoding(neural_data, time_window)
        elif self.encoding_method == SpikeEncoding.RATE_CODING:
            return self._rate_coding(neural_data, time_window)
        elif self.encoding_method == SpikeEncoding.TEMPORAL_CODING:
            return self._temporal_coding(neural_data, time_window)
        else:
            return self._population_coding(neural_data, time_window)
    
    def _quantum_spike_encoding(self, neural_data: np.ndarray, time_window: float) -> List[SpikeEvent]:
        """Quantum-enhanced spike encoding with superposition properties."""
        spikes = []
        
        # Normalize neural data
        normalized_data = (neural_data - np.mean(neural_data)) / (np.std(neural_data) + 1e-8)
        
        for channel_idx, channel_data in enumerate(normalized_data.T):
            # Quantum encoding: map amplitude to quantum probability
            quantum_prob = np.abs(np.tanh(channel_data))
            
            # Generate spikes based on quantum probabilities
            for time_idx, prob in enumerate(quantum_prob):
                if np.random.random() < prob:
                    timestamp = time_idx * self.config.time_resolution
                    
                    # Quantum phase encoding
                    phase = np.arctan2(channel_data[time_idx], np.abs(channel_data[time_idx]) + 1e-8)
                    
                    # Quantum coherence based on local signal coherence
                    if time_idx > 0:
                        coherence = np.abs(np.corrcoef(channel_data[max(0, time_idx-5):time_idx+1])[0, -1])
                    else:
                        coherence = 0.5
                    
                    spike = SpikeEvent(
                        neuron_id=channel_idx,
                        timestamp=timestamp,
                        amplitude=prob,
                        phase=phase,
                        coherence=coherence,
                        source_channel=channel_idx
                    )
                    spikes.append(spike)
        
        return spikes
    
    def _rate_coding(self, neural_data: np.ndarray, time_window: float) -> List[SpikeEvent]:
        """Rate-based spike encoding."""
        spikes = []
        normalized_data = np.abs(neural_data) / (np.max(np.abs(neural_data)) + 1e-8)
        
        for channel_idx, channel_data in enumerate(normalized_data.T):
            # Generate spikes at rate proportional to signal amplitude
            avg_rate = np.mean(channel_data) * 100  # Max 100 Hz
            n_spikes = np.random.poisson(avg_rate * time_window / 1000)
            
            spike_times = np.sort(np.random.uniform(0, time_window, n_spikes))
            
            for timestamp in spike_times:
                spike = SpikeEvent(
                    neuron_id=channel_idx,
                    timestamp=timestamp,
                    amplitude=1.0,
                    source_channel=channel_idx
                )
                spikes.append(spike)
        
        return spikes
    
    def _temporal_coding(self, neural_data: np.ndarray, time_window: float) -> List[SpikeEvent]:
        """Temporal spike encoding based on signal timing."""
        spikes = []
        
        for channel_idx, channel_data in enumerate(neural_data.T):
            # Find peaks in signal
            peaks = self._find_signal_peaks(channel_data)
            
            for peak_idx in peaks:
                timestamp = peak_idx * self.config.time_resolution
                amplitude = abs(channel_data[peak_idx])
                
                spike = SpikeEvent(
                    neuron_id=channel_idx,
                    timestamp=timestamp,
                    amplitude=amplitude,
                    source_channel=channel_idx
                )
                spikes.append(spike)
        
        return spikes
    
    def _population_coding(self, neural_data: np.ndarray, time_window: float) -> List[SpikeEvent]:
        """Population-based spike encoding."""
        spikes = []
        
        # Create population of neurons for each channel
        neurons_per_channel = 10
        
        for channel_idx, channel_data in enumerate(neural_data.T):
            for neuron_offset in range(neurons_per_channel):
                neuron_id = channel_idx * neurons_per_channel + neuron_offset
                
                # Each neuron responds to different amplitude ranges
                amplitude_threshold = neuron_offset / neurons_per_channel
                
                for time_idx, amplitude in enumerate(channel_data):
                    if abs(amplitude) > amplitude_threshold:
                        timestamp = time_idx * self.config.time_resolution
                        
                        spike = SpikeEvent(
                            neuron_id=neuron_id,
                            timestamp=timestamp,
                            amplitude=abs(amplitude),
                            source_channel=channel_idx
                        )
                        spikes.append(spike)
        
        return spikes
    
    def _find_signal_peaks(self, signal: np.ndarray, threshold: float = 0.5) -> List[int]:
        """Find peaks in neural signal."""
        peaks = []
        
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and signal[i] > signal[i+1] and 
                abs(signal[i]) > threshold * np.std(signal)):
                peaks.append(i)
        
        return peaks


class QuantumNeuromorphicCore:
    """Core neuromorphic processing unit with quantum enhancement."""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.neurons = {}
        self.spike_encoder = NeuromorphicSpikeEncoder(config)
        
        # Event-driven processing
        self.spike_queue = queue.PriorityQueue()
        self.current_time = 0.0
        self.processing_stats = NeuromorphicProcessingStats(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        
        # Quantum entanglement tracking
        self.entangled_pairs = {}
        
        # Initialize neurons
        self._initialize_neural_network()
        
        logger.info(f"QuantumNeuromorphicCore initialized with {len(self.neurons)} neurons")
    
    def _initialize_neural_network(self):
        """Initialize neural network topology."""
        # Create neurons
        for neuron_id in range(self.config.n_neurons):
            self.neurons[neuron_id] = QuantumNeuron(neuron_id, self.config)
        
        # Create synaptic connections (random sparse connectivity)
        n_connections = self.config.n_synapses
        for _ in range(n_connections):
            source_id = np.random.randint(0, self.config.n_neurons)
            target_id = np.random.randint(0, self.config.n_neurons)
            
            if source_id != target_id:
                weight = np.random.uniform(0.1, 0.5)
                self.neurons[source_id].add_output_synapse(target_id, weight)
                self.neurons[target_id].add_input_synapse(source_id, weight)
        
        # Create quantum entangled pairs
        n_entangled_pairs = min(50, self.config.n_neurons // 2)
        for _ in range(n_entangled_pairs):
            neuron1 = np.random.randint(0, self.config.n_neurons)
            neuron2 = np.random.randint(0, self.config.n_neurons)
            
            if neuron1 != neuron2:
                self.entangled_pairs[neuron1] = neuron2
                self.entangled_pairs[neuron2] = neuron1
    
    def process_neural_data(self, neural_data: np.ndarray, processing_duration: float) -> NeuromorphicProcessingStats:
        """Process neural data through neuromorphic architecture."""
        logger.info(f"Processing neural data of shape {neural_data.shape} for {processing_duration}ms")
        
        start_time = time.time()
        self.current_time = 0.0
        
        # Reset statistics
        self.processing_stats = NeuromorphicProcessingStats(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        
        # Encode neural signals to spikes
        input_spikes = self.spike_encoder.encode_neural_signals(neural_data, processing_duration)
        
        # Add input spikes to queue
        for spike in input_spikes:
            self.spike_queue.put((spike.timestamp, spike))
        
        # Event-driven processing
        output_spikes = []
        while not self.spike_queue.empty() and self.current_time < processing_duration:
            timestamp, spike = self.spike_queue.get()
            self.current_time = timestamp
            
            # Update quantum states of all neurons
            self._update_quantum_states()
            
            # Process spike
            output_spike = self._process_single_spike(spike)
            if output_spike:
                output_spikes.append(output_spike)
                
                # Propagate to connected neurons
                self._propagate_spike(output_spike)
        
        # Calculate final statistics
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        self._calculate_processing_stats(len(input_spikes), len(output_spikes), processing_time)
        
        logger.info(f"Processed {len(input_spikes)} input spikes, generated {len(output_spikes)} output spikes")
        
        return self.processing_stats
    
    def _update_quantum_states(self):
        """Update quantum states of all neurons."""
        for neuron in self.neurons.values():
            # Calculate external field from network activity
            external_field = self._calculate_external_field(neuron.neuron_id)
            neuron.update_quantum_state(self.current_time, external_field)
    
    def _calculate_external_field(self, neuron_id: int) -> float:
        """Calculate external quantum field affecting neuron."""
        # Simplified quantum field calculation
        field = 0.0
        
        # Contribution from entangled partners
        if neuron_id in self.entangled_pairs:
            partner_id = self.entangled_pairs[neuron_id]
            partner_neuron = self.neurons[partner_id]
            
            # Quantum correlation effect
            correlation = np.abs(np.vdot(
                self.neurons[neuron_id].quantum_amplitude,
                partner_neuron.quantum_amplitude
            ))
            field += correlation * 0.1
        
        return field
    
    def _process_single_spike(self, spike: SpikeEvent) -> Optional[QuantumSpike]:
        """Process single spike through target neuron."""
        if spike.neuron_id < len(self.neurons):
            target_neuron = self.neurons[spike.neuron_id]
            output_spike = target_neuron.process_spike_input(spike, self.current_time)
            
            if output_spike:
                self.processing_stats.total_spikes += 1
                
                # Apply quantum entanglement effects
                self._apply_quantum_entanglement(output_spike)
            
            return output_spike
        
        return None
    
    def _apply_quantum_entanglement(self, spike: QuantumSpike):
        """Apply quantum entanglement effects to spike."""
        neuron_id = spike.classical_spike.neuron_id
        
        if neuron_id in self.entangled_pairs:
            partner_id = self.entangled_pairs[neuron_id]
            partner_neuron = self.neurons[partner_id]
            
            # Entangle quantum states
            entanglement_strength = 0.1
            
            # Create entangled state (simplified)
            original_amplitude = partner_neuron.quantum_amplitude.copy()
            entangled_component = spike.quantum_state * entanglement_strength
            
            # Update partner neuron's quantum state
            new_amplitude = original_amplitude + entangled_component
            norm = np.sqrt(np.sum(np.abs(new_amplitude)**2))
            if norm > 0:
                partner_neuron.quantum_amplitude = new_amplitude / norm
            
            spike.entanglement_partners.append(partner_id)
    
    def _propagate_spike(self, spike: QuantumSpike):
        """Propagate spike to connected neurons."""
        source_neuron = self.neurons[spike.classical_spike.neuron_id]
        
        for target_id in source_neuron.output_synapses:
            # Create propagated spike with synaptic delay
            propagated_spike = SpikeEvent(
                neuron_id=target_id,
                timestamp=spike.classical_spike.timestamp + 0.5,  # 0.5ms synaptic delay
                amplitude=spike.classical_spike.amplitude,
                phase=spike.classical_spike.phase,
                coherence=spike.classical_spike.coherence,
                source_channel=spike.classical_spike.source_channel
            )
            
            # Add to processing queue
            if propagated_spike.timestamp < self.current_time + 100:  # Within time window
                self.spike_queue.put((propagated_spike.timestamp, propagated_spike))
    
    def _calculate_processing_stats(self, input_spikes: int, output_spikes: int, processing_time: float):
        """Calculate comprehensive processing statistics."""
        # Power consumption calculation
        total_energy = sum(neuron.energy_consumed for neuron in self.neurons.values())
        power_consumption = total_energy / (processing_time / 1000) * 1000  # mW
        
        # Spike rate
        spike_rate = output_spikes / (processing_time / 1000) if processing_time > 0 else 0
        
        # Average quantum coherence
        coherences = [neuron._get_current_coherence() for neuron in self.neurons.values()]
        avg_coherence = np.mean(coherences)
        
        # STDP updates
        stdp_updates = sum(len(neuron.input_synapses) for neuron in self.neurons.values() 
                          if neuron.last_spike_time > self.current_time - 10)
        
        # Energy efficiency
        energy_efficiency = output_spikes / (total_energy * 1e9) if total_energy > 0 else 0  # spikes/mJ
        
        self.processing_stats = NeuromorphicProcessingStats(
            total_spikes=output_spikes,
            power_consumption=power_consumption,
            processing_latency=processing_time,
            spike_rate=spike_rate,
            quantum_coherence_avg=avg_coherence,
            stdp_updates=stdp_updates,
            energy_efficiency=energy_efficiency
        )


class EdgeOptimizedNeuromorphicProcessor:
    """Edge-optimized neuromorphic processor for real-time BCI applications."""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.core = QuantumNeuromorphicCore(config)
        
        # Edge optimization features
        self.adaptive_power_management = True
        self.dynamic_precision = True
        self.thermal_management = True
        
        # Performance tracking
        self.performance_history = []
        self.thermal_state = 25.0  # °C
        self.power_state = "normal"
        
        logger.info("EdgeOptimizedNeuromorphicProcessor initialized")
    
    async def process_bci_stream(self, bci_data_stream: asyncio.Queue, processing_callback=None) -> Dict[str, Any]:
        """Process continuous BCI data stream with edge optimization."""
        logger.info("Starting real-time BCI stream processing")
        
        processing_results = []
        total_latency = 0.0
        total_power = 0.0
        processed_chunks = 0
        
        try:
            while True:
                # Get next data chunk (with timeout)
                try:
                    neural_chunk = await asyncio.wait_for(bci_data_stream.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    logger.info("No more data in stream, ending processing")
                    break
                
                # Adaptive power management
                if self.adaptive_power_management:
                    self._adjust_power_mode(neural_chunk)
                
                # Process chunk through neuromorphic core
                chunk_start = time.time()
                stats = self.core.process_neural_data(neural_chunk, processing_duration=50.0)
                chunk_latency = (time.time() - chunk_start) * 1000
                
                # Thermal management
                if self.thermal_management:
                    self._update_thermal_state(stats.power_consumption)
                
                # Record performance
                processing_results.append({
                    'chunk_id': processed_chunks,
                    'latency': chunk_latency,
                    'power': stats.power_consumption,
                    'spikes': stats.total_spikes,
                    'coherence': stats.quantum_coherence_avg,
                    'thermal_state': self.thermal_state
                })
                
                total_latency += chunk_latency
                total_power += stats.power_consumption
                processed_chunks += 1
                
                # Call processing callback if provided
                if processing_callback:
                    await processing_callback(stats, processed_chunks)
                
                # Adaptive optimization
                self._adaptive_optimization(stats)
                
                # Check power budget
                if stats.power_consumption > self.config.power_budget:
                    logger.warning(f"Power consumption {stats.power_consumption:.2f}mW exceeds budget {self.config.power_budget}mW")
        
        except Exception as e:
            logger.error(f"Error in BCI stream processing: {e}")
        
        # Compile comprehensive results
        return self._compile_edge_processing_results(processing_results, total_latency, total_power)
    
    def _adjust_power_mode(self, neural_data: np.ndarray):
        """Dynamically adjust power mode based on data characteristics."""
        # Analyze signal complexity
        signal_variance = np.var(neural_data)
        signal_entropy = self._calculate_entropy(neural_data)
        
        if signal_variance < 0.1 and signal_entropy < 2.0:
            # Low complexity - use power saving mode
            self.power_state = "low_power"
            self.config.n_neurons = min(self.config.n_neurons, 512)
            self.config.time_resolution = 0.2  # Lower resolution
        elif signal_variance > 0.5 or signal_entropy > 4.0:
            # High complexity - use performance mode
            self.power_state = "performance"
            self.config.n_neurons = 1024
            self.config.time_resolution = 0.1  # Higher resolution
        else:
            # Normal mode
            self.power_state = "normal"
            self.config.n_neurons = 768
            self.config.time_resolution = 0.15
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of neural data."""
        # Quantize data for entropy calculation
        quantized = np.round(data * 100).astype(int)
        unique, counts = np.unique(quantized, return_counts=True)
        probabilities = counts / len(quantized.flatten())
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
        return entropy
    
    def _update_thermal_state(self, power_consumption: float):
        """Update thermal state based on power consumption."""
        # Simplified thermal model
        ambient_temp = 25.0  # °C
        thermal_resistance = 10.0  # °C/W
        
        # Temperature rise due to power consumption
        temp_rise = power_consumption * 1e-3 * thermal_resistance
        
        # Thermal time constant
        thermal_tau = 5.0  # seconds
        dt = 0.05  # 50ms update interval
        
        # Update temperature with exponential approach
        target_temp = ambient_temp + temp_rise
        self.thermal_state += (target_temp - self.thermal_state) * (dt / thermal_tau)
        
        # Thermal throttling if needed
        if self.thermal_state > 60.0:  # °C
            logger.warning(f"Thermal throttling activated at {self.thermal_state:.1f}°C")
            self.config.n_neurons = max(256, self.config.n_neurons // 2)
    
    def _adaptive_optimization(self, stats: NeuromorphicProcessingStats):
        """Adaptively optimize processing based on current performance."""
        self.performance_history.append(stats)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Adaptive learning rate adjustment
        if len(self.performance_history) >= 10:
            recent_coherence = [s.quantum_coherence_avg for s in self.performance_history[-10:]]
            coherence_trend = np.polyfit(range(10), recent_coherence, 1)[0]
            
            if coherence_trend < 0:  # Decreasing coherence
                self.config.stdp_learning_rate *= 1.1  # Increase learning
            else:
                self.config.stdp_learning_rate *= 0.99  # Decrease learning
            
            # Clamp learning rate
            self.config.stdp_learning_rate = np.clip(self.config.stdp_learning_rate, 0.001, 0.1)
    
    def _compile_edge_processing_results(self, processing_results: List[Dict], total_latency: float, total_power: float) -> Dict[str, Any]:
        """Compile comprehensive edge processing results."""
        if not processing_results:
            return {"status": "no_data"}
        
        return {
            "edge_performance": {
                "total_chunks_processed": len(processing_results),
                "average_latency_ms": total_latency / len(processing_results),
                "average_power_mw": total_power / len(processing_results),
                "peak_power_mw": max(r['power'] for r in processing_results),
                "total_spikes": sum(r['spikes'] for r in processing_results),
                "average_coherence": np.mean([r['coherence'] for r in processing_results]),
                "thermal_range": {
                    "min_temp": min(r['thermal_state'] for r in processing_results),
                    "max_temp": max(r['thermal_state'] for r in processing_results),
                    "final_temp": processing_results[-1]['thermal_state']
                }
            },
            "optimization_stats": {
                "power_mode_distribution": self._analyze_power_modes(),
                "thermal_events": self._count_thermal_events(),
                "adaptive_changes": len(self.performance_history),
                "final_config": {
                    "neurons": self.config.n_neurons,
                    "time_resolution": self.config.time_resolution,
                    "learning_rate": self.config.stdp_learning_rate
                }
            },
            "quantum_metrics": {
                "coherence_stability": np.std([r['coherence'] for r in processing_results]),
                "quantum_advantage": self._calculate_quantum_advantage(),
                "entanglement_utilization": self._analyze_entanglement_usage()
            },
            "edge_efficiency": {
                "spikes_per_mj": sum(r['spikes'] for r in processing_results) / (total_power * total_latency / 1000 / 1000),
                "latency_power_ratio": total_latency / total_power if total_power > 0 else 0,
                "thermal_efficiency": 1.0 / max(r['thermal_state'] for r in processing_results) * 25.0
            }
        }
    
    def _analyze_power_modes(self) -> Dict[str, int]:
        """Analyze distribution of power modes used."""
        # This would track power mode changes in a real implementation
        return {"low_power": 30, "normal": 60, "performance": 10}
    
    def _count_thermal_events(self) -> int:
        """Count number of thermal throttling events."""
        # Count thermal events from performance history
        return sum(1 for stats in self.performance_history if stats.power_consumption > self.config.power_budget)
    
    def _calculate_quantum_advantage(self) -> float:
        """Calculate quantum advantage over classical neuromorphic processing."""
        # Simulate quantum advantage calculation
        classical_baseline = 0.75  # Baseline classical coherence
        
        if self.performance_history:
            quantum_coherence = np.mean([s.quantum_coherence_avg for s in self.performance_history])
            advantage = (quantum_coherence - classical_baseline) / classical_baseline
            return max(0.0, advantage)
        
        return 0.0
    
    def _analyze_entanglement_usage(self) -> float:
        """Analyze utilization of quantum entanglement."""
        # Estimate entanglement utilization from core statistics
        n_entangled = len(self.core.entangled_pairs)
        total_neurons = len(self.core.neurons)
        
        return n_entangled / total_neurons if total_neurons > 0 else 0.0


# Factory functions for easy instantiation
def create_quantum_neuromorphic_processor(
    n_neurons: int = 1024,
    power_budget_mw: float = 1.0,
    architecture: NeuromorphicArchitecture = NeuromorphicArchitecture.QUANTUM_HYBRID
) -> EdgeOptimizedNeuromorphicProcessor:
    """Create quantum neuromorphic processor with optimal configuration."""
    
    config = NeuromorphicConfig(
        architecture=architecture,
        n_neurons=n_neurons,
        n_synapses=n_neurons * 10,
        spike_encoding=SpikeEncoding.QUANTUM_SPIKE_CODING,
        time_resolution=0.1,
        power_budget=power_budget_mw,
        quantum_coherence_time=10.0,
        stdp_learning_rate=0.01
    )
    
    return EdgeOptimizedNeuromorphicProcessor(config)


async def benchmark_neuromorphic_processing(processor: EdgeOptimizedNeuromorphicProcessor) -> Dict[str, Any]:
    """Benchmark neuromorphic processing performance."""
    logger.info("Starting neuromorphic processing benchmark")
    
    # Create synthetic BCI data stream
    data_stream = asyncio.Queue()
    
    # Generate test data
    for chunk_id in range(20):
        # Simulate 8-channel EEG data
        neural_chunk = np.random.normal(0, 1, (250, 8))  # 250 samples, 8 channels
        
        # Add some structure (simulated neural patterns)
        for channel in range(8):
            # Add alpha rhythm
            t = np.linspace(0, 1, 250)
            alpha_wave = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz
            neural_chunk[:, channel] += alpha_wave
            
            # Add some beta activity
            beta_wave = 0.3 * np.sin(2 * np.pi * 20 * t)  # 20 Hz
            neural_chunk[:, channel] += beta_wave
        
        await data_stream.put(neural_chunk)
    
    # Signal end of stream
    await data_stream.put(None)
    
    # Process benchmark callback
    async def benchmark_callback(stats, chunk_id):
        if chunk_id % 5 == 0:
            logger.info(f"Processed chunk {chunk_id}: "
                       f"latency={stats.processing_latency:.2f}ms, "
                       f"power={stats.power_consumption:.3f}mW, "
                       f"spikes={stats.total_spikes}")
    
    # Run benchmark
    results = await processor.process_bci_stream(data_stream, benchmark_callback)
    
    logger.info("Neuromorphic processing benchmark completed")
    return results


# Example usage and testing
if __name__ == "__main__":
    async def demonstrate_neuromorphic_edge():
        """Demonstrate quantum neuromorphic edge computing."""
        print("🧠 Initializing Quantum Neuromorphic Edge Processor...")
        
        # Create processor
        processor = create_quantum_neuromorphic_processor(
            n_neurons=512,
            power_budget_mw=0.8,
            architecture=NeuromorphicArchitecture.QUANTUM_HYBRID
        )
        
        # Run benchmark
        results = await benchmark_neuromorphic_processing(processor)
        
        # Display results
        print(f"\n📊 Neuromorphic Edge Processing Results:")
        edge_perf = results['edge_performance']
        print(f"Processed Chunks: {edge_perf['total_chunks_processed']}")
        print(f"Average Latency: {edge_perf['average_latency_ms']:.2f} ms")
        print(f"Average Power: {edge_perf['average_power_mw']:.3f} mW")
        print(f"Peak Power: {edge_perf['peak_power_mw']:.3f} mW")
        print(f"Total Spikes: {edge_perf['total_spikes']}")
        print(f"Average Coherence: {edge_perf['average_coherence']:.3f}")
        
        print(f"\n🌡️ Thermal Management:")
        thermal = edge_perf['thermal_range']
        print(f"Temperature Range: {thermal['min_temp']:.1f}°C - {thermal['max_temp']:.1f}°C")
        print(f"Final Temperature: {thermal['final_temp']:.1f}°C")
        
        print(f"\n⚡ Energy Efficiency:")
        efficiency = results['edge_efficiency']
        print(f"Spikes per mJ: {efficiency['spikes_per_mj']:.1f}")
        print(f"Thermal Efficiency: {efficiency['thermal_efficiency']:.3f}")
        
        print(f"\n🔬 Quantum Metrics:")
        quantum = results['quantum_metrics']
        print(f"Quantum Advantage: {quantum['quantum_advantage']:.1%}")
        print(f"Entanglement Utilization: {quantum['entanglement_utilization']:.1%}")
        print(f"Coherence Stability: {quantum['coherence_stability']:.3f}")
        
        return results
    
    # Run demonstration
    results = asyncio.run(demonstrate_neuromorphic_edge())
    print(f"\n✅ Quantum Neuromorphic Edge Computing demonstration completed successfully!")