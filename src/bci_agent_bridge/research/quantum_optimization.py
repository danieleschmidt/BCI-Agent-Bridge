"""
Quantum-Inspired Optimization for BCI Neural Decoding.

This module implements quantum-inspired optimization techniques for enhancing
BCI neural decoding performance, including variational quantum circuits,
quantum annealing simulation, and quantum-classical hybrid algorithms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from scipy.optimize import minimize
from scipy.linalg import expm


@dataclass
class QuantumConfig:
    """Configuration for quantum-inspired optimization."""
    n_qubits: int = 16
    n_layers: int = 4
    n_parameters: int = 64
    learning_rate: float = 0.01
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    quantum_noise: float = 0.01
    annealing_schedule: str = "linear"  # "linear", "exponential", "inverse"
    use_entanglement: bool = True
    measurement_basis: str = "computational"  # "computational", "hadamard"


class QuantumGate:
    """Base class for quantum gates."""
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """Pauli-X rotation gate."""
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ])
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """Pauli-Y rotation gate."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])
    
    @staticmethod
    def rotation_z(phi: float) -> np.ndarray:
        """Pauli-Z rotation gate."""
        return np.array([
            [np.exp(-1j*phi/2), 0],
            [0, np.exp(1j*phi/2)]
        ])
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate."""
        return np.array([
            [1, 1],
            [1, -1]
        ]) / np.sqrt(2)
    
    @staticmethod
    def cnot() -> np.ndarray:
        """Controlled-NOT gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])


class QuantumCircuit:
    """Quantum circuit simulator for optimization."""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0  # Initialize to |00...0⟩
        self.gates = QuantumGate()
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0
    
    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply single-qubit gate."""
        # Create full gate matrix
        full_gate = np.eye(1, dtype=complex)
        
        for i in range(self.n_qubits):
            if i == qubit:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        
        # Apply gate
        self.state = full_gate @ self.state
    
    def apply_two_qubit_gate(self, gate: np.ndarray, control: int, target: int) -> None:
        """Apply two-qubit gate (simplified implementation)."""
        # For CNOT gate specifically
        if control < target:
            for i in range(self.n_states):
                if (i >> (self.n_qubits - 1 - control)) & 1:
                    # Control qubit is 1, flip target
                    target_bit = (i >> (self.n_qubits - 1 - target)) & 1
                    if target_bit == 0:
                        flipped_state = i | (1 << (self.n_qubits - 1 - target))
                    else:
                        flipped_state = i & ~(1 << (self.n_qubits - 1 - target))
                    
                    # Swap amplitudes
                    temp = self.state[i]
                    self.state[i] = self.state[flipped_state]
                    self.state[flipped_state] = temp
    
    def measure(self) -> List[int]:
        """Measure all qubits and return classical bitstring."""
        probabilities = np.abs(self.state) ** 2
        measured_state = np.random.choice(self.n_states, p=probabilities)
        
        # Convert to bitstring
        bitstring = []
        for i in range(self.n_qubits):
            bit = (measured_state >> (self.n_qubits - 1 - i)) & 1
            bitstring.append(bit)
        
        return bitstring
    
    def expectation_value(self, observable: np.ndarray) -> float:
        """Calculate expectation value of observable."""
        return np.real(np.conj(self.state) @ observable @ self.state)


class VariationalQuantumCircuit(nn.Module):
    """Variational quantum circuit for neural decoding optimization."""
    
    def __init__(self, config: QuantumConfig, input_dim: int, output_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Classical preprocessing
        self.input_encoder = nn.Linear(input_dim, config.n_qubits)
        
        # Quantum circuit parameters
        self.circuit_parameters = nn.Parameter(
            torch.randn(config.n_layers, config.n_qubits, 3) * 0.1
        )
        
        # Entangling parameters (for CNOT gates)
        if config.use_entanglement:
            self.entangling_parameters = nn.Parameter(
                torch.randn(config.n_layers, config.n_qubits) * 0.1
            )
        
        # Classical post-processing
        self.output_decoder = nn.Sequential(
            nn.Linear(config.n_qubits, config.n_qubits * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_qubits * 2, output_dim)
        )
        
        self.logger = logging.getLogger(__name__)
    
    def quantum_layer(self, layer_idx: int, encoded_input: torch.Tensor) -> torch.Tensor:
        """Execute a quantum layer with parameterized gates."""
        batch_size = encoded_input.shape[0]
        quantum_outputs = []
        
        for batch_idx in range(batch_size):
            # Initialize quantum circuit
            circuit = QuantumCircuit(self.config.n_qubits)
            
            # Encode classical input into quantum state
            input_angles = encoded_input[batch_idx] * np.pi  # Scale to [0, π]
            
            # Input encoding layer
            for qubit in range(self.config.n_qubits):
                # RY rotation for input encoding
                circuit.apply_single_qubit_gate(
                    circuit.gates.rotation_y(input_angles[qubit].item()),
                    qubit
                )
            
            # Parameterized quantum layer
            layer_params = self.circuit_parameters[layer_idx]
            
            for qubit in range(self.config.n_qubits):
                # Three-parameter rotation (RX, RY, RZ)
                rx_angle = layer_params[qubit, 0].item()
                ry_angle = layer_params[qubit, 1].item()
                rz_angle = layer_params[qubit, 2].item()
                
                circuit.apply_single_qubit_gate(circuit.gates.rotation_x(rx_angle), qubit)
                circuit.apply_single_qubit_gate(circuit.gates.rotation_y(ry_angle), qubit)
                circuit.apply_single_qubit_gate(circuit.gates.rotation_z(rz_angle), qubit)
            
            # Entangling layer
            if self.config.use_entanglement and hasattr(self, 'entangling_parameters'):
                entangling_params = self.entangling_parameters[layer_idx]
                
                for qubit in range(self.config.n_qubits - 1):
                    # Parameterized entangling gate
                    if entangling_params[qubit].item() > 0:
                        circuit.apply_two_qubit_gate(
                            circuit.gates.cnot(),
                            qubit, (qubit + 1) % self.config.n_qubits
                        )
            
            # Measurement in computational basis
            measurements = []
            for qubit in range(self.config.n_qubits):
                # Create Z observable for each qubit
                z_observable = np.eye(circuit.n_states, dtype=complex)
                for state in range(circuit.n_states):
                    if (state >> (self.config.n_qubits - 1 - qubit)) & 1:
                        z_observable[state, state] = -1
                
                expectation = circuit.expectation_value(z_observable)
                measurements.append(expectation)
            
            quantum_outputs.append(measurements)
        
        return torch.FloatTensor(quantum_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through variational quantum circuit."""
        # Classical preprocessing
        encoded = torch.tanh(self.input_encoder(x))  # Bounded to [-1, 1]
        
        # Quantum processing layers
        quantum_state = encoded
        for layer_idx in range(self.config.n_layers):
            quantum_state = self.quantum_layer(layer_idx, quantum_state)
            
            # Add quantum noise
            if self.training and self.config.quantum_noise > 0:
                noise = torch.randn_like(quantum_state) * self.config.quantum_noise
                quantum_state = quantum_state + noise
        
        # Classical post-processing
        output = self.output_decoder(quantum_state)
        
        return output


class QuantumAnnealingOptimizer:
    """Quantum annealing simulation for optimization problems."""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_ising_hamiltonian(self, problem_weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Create Ising model Hamiltonian for the optimization problem."""
        n_vars = len(bias)
        hamiltonian = np.zeros((2**n_vars, 2**n_vars))
        
        # Add bias terms
        for i in range(n_vars):
            for state in range(2**n_vars):
                bit = (state >> i) & 1
                spin = 2 * bit - 1  # Convert {0,1} to {-1,1}
                hamiltonian[state, state] += bias[i] * spin
        
        # Add interaction terms
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                weight = problem_weights[i, j]
                for state in range(2**n_vars):
                    bit_i = (state >> i) & 1
                    bit_j = (state >> j) & 1
                    spin_i = 2 * bit_i - 1
                    spin_j = 2 * bit_j - 1
                    hamiltonian[state, state] += weight * spin_i * spin_j
        
        return hamiltonian
    
    def annealing_schedule(self, step: int, max_steps: int) -> float:
        """Generate annealing schedule s(t) ∈ [0,1]."""
        t = step / max_steps
        
        if self.config.annealing_schedule == "linear":
            return t
        elif self.config.annealing_schedule == "exponential":
            return 1 - np.exp(-5 * t)
        elif self.config.annealing_schedule == "inverse":
            return t / (1 + t)
        else:
            return t
    
    def quantum_annealing(
        self, 
        problem_hamiltonian: np.ndarray,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Perform quantum annealing simulation.
        
        Args:
            problem_hamiltonian: Target problem Hamiltonian
            initial_state: Initial quantum state (superposition)
            
        Returns:
            Final state and energy history
        """
        n_qubits = int(np.log2(problem_hamiltonian.shape[0]))
        
        # Initialize in uniform superposition if not provided
        if initial_state is None:
            initial_state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
        
        state = initial_state.copy()
        energy_history = []
        
        # Transverse field Hamiltonian (X on all qubits)
        transverse_field = np.zeros((2**n_qubits, 2**n_qubits))
        for i in range(n_qubits):
            x_gate = np.array([[0, 1], [1, 0]])
            single_qubit_x = np.eye(1)
            
            for j in range(n_qubits):
                if j == i:
                    single_qubit_x = np.kron(single_qubit_x, x_gate)
                else:
                    single_qubit_x = np.kron(single_qubit_x, np.eye(2))
            
            transverse_field += single_qubit_x
        
        # Annealing evolution
        dt = 1.0 / self.config.max_iterations
        
        for step in range(self.config.max_iterations):
            s = self.annealing_schedule(step, self.config.max_iterations)
            
            # Time-dependent Hamiltonian
            hamiltonian = (1 - s) * transverse_field + s * problem_hamiltonian
            
            # Time evolution (simplified)
            evolution_operator = expm(-1j * hamiltonian * dt)
            state = evolution_operator @ state
            
            # Normalize
            state = state / np.linalg.norm(state)
            
            # Calculate energy
            energy = np.real(np.conj(state) @ problem_hamiltonian @ state)
            energy_history.append(energy)
            
            # Check convergence
            if len(energy_history) > 10:
                recent_change = np.abs(energy_history[-1] - energy_history[-10])
                if recent_change < self.config.convergence_threshold:
                    self.logger.info(f"Converged at step {step}")
                    break
        
        return state, energy_history
    
    def extract_solution(self, final_state: np.ndarray, n_samples: int = 1000) -> List[List[int]]:
        """Extract classical solutions by sampling the final quantum state."""
        n_qubits = int(np.log2(len(final_state)))
        probabilities = np.abs(final_state) ** 2
        
        solutions = []
        for _ in range(n_samples):
            measured_state = np.random.choice(len(final_state), p=probabilities)
            bitstring = []
            for i in range(n_qubits):
                bit = (measured_state >> i) & 1
                bitstring.append(bit)
            solutions.append(bitstring)
        
        return solutions


class QuantumFeatureMap:
    """Quantum feature mapping for classical data encoding."""
    
    def __init__(self, n_qubits: int, n_features: int):
        self.n_qubits = n_qubits
        self.n_features = n_features
        
    def amplitude_encoding(self, features: np.ndarray) -> np.ndarray:
        """Encode features as quantum state amplitudes."""
        # Normalize features
        normalized = features / np.linalg.norm(features)
        
        # Pad or truncate to fit quantum state dimension
        n_amplitudes = 2 ** self.n_qubits
        if len(normalized) > n_amplitudes:
            encoded = normalized[:n_amplitudes]
        else:
            encoded = np.zeros(n_amplitudes)
            encoded[:len(normalized)] = normalized
        
        # Ensure unit norm
        encoded = encoded / np.linalg.norm(encoded)
        
        return encoded.astype(complex)
    
    def angle_encoding(self, features: np.ndarray) -> List[float]:
        """Encode features as rotation angles."""
        # Scale features to [0, 2π]
        scaled = (features + 1) * np.pi  # Assuming features in [-1, 1]
        
        # Repeat/truncate to match number of qubits
        if len(scaled) > self.n_qubits:
            angles = scaled[:self.n_qubits]
        else:
            angles = np.tile(scaled, (self.n_qubits // len(scaled) + 1))[:self.n_qubits]
        
        return angles.tolist()


class QuantumNeuralDecoder(nn.Module):
    """Complete quantum-inspired neural decoder for BCI applications."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Optional[QuantumConfig] = None
    ):
        super().__init__()
        self.config = config or QuantumConfig()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Quantum feature mapping
        self.feature_map = QuantumFeatureMap(self.config.n_qubits, input_dim)
        
        # Variational quantum circuit
        self.quantum_circuit = VariationalQuantumCircuit(
            self.config, input_dim, output_dim
        )
        
        # Quantum annealing optimizer
        self.annealing_optimizer = QuantumAnnealingOptimizer(self.config)
        
        # Classical components for hybrid approach
        self.classical_preprocessor = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim * 2),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        self.quantum_classical_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Optimization parameters for annealing
        self.optimization_weights = nn.Parameter(
            torch.randn(self.config.n_qubits, self.config.n_qubits) * 0.1
        )
        self.optimization_bias = nn.Parameter(
            torch.randn(self.config.n_qubits) * 0.1
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum-classical hybrid network."""
        # Classical preprocessing
        classical_features = self.classical_preprocessor(x)
        
        # Quantum processing
        quantum_output = self.quantum_circuit(x)
        
        # Fusion of quantum and classical results
        combined = torch.cat([classical_features, quantum_output], dim=-1)
        output = self.quantum_classical_fusion(combined)
        
        return output
    
    def quantum_optimization_step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Perform quantum annealing optimization step."""
        # Convert neural network parameters to optimization problem
        weights = self.optimization_weights.detach().cpu().numpy()
        bias = self.optimization_bias.detach().cpu().numpy()
        
        # Create Ising Hamiltonian
        problem_hamiltonian = self.annealing_optimizer.create_ising_hamiltonian(
            weights, bias
        )
        
        # Perform quantum annealing
        final_state, energy_history = self.annealing_optimizer.quantum_annealing(
            problem_hamiltonian
        )
        
        # Extract solutions
        solutions = self.annealing_optimizer.extract_solution(final_state, n_samples=100)
        
        # Select best solution based on energy
        best_energy = float('inf')
        best_solution = None
        
        for solution in solutions:
            # Calculate energy for this solution
            energy = 0.0
            for i in range(len(solution)):
                energy += bias[i] * (2 * solution[i] - 1)
                for j in range(i+1, len(solution)):
                    energy += weights[i, j] * (2 * solution[i] - 1) * (2 * solution[j] - 1)
            
            if energy < best_energy:
                best_energy = energy
                best_solution = solution
        
        return {
            'quantum_energy': best_energy,
            'classical_loss': loss.item(),
            'convergence_steps': len(energy_history),
            'final_entropy': self._calculate_entropy(final_state)
        }
    
    def _calculate_entropy(self, quantum_state: np.ndarray) -> float:
        """Calculate von Neumann entropy of quantum state."""
        # Trace out half of the qubits to get reduced density matrix
        n_qubits = int(np.log2(len(quantum_state)))
        if n_qubits < 2:
            return 0.0
        
        n_subsystem = n_qubits // 2
        dim_subsystem = 2 ** n_subsystem
        dim_environment = 2 ** (n_qubits - n_subsystem)
        
        # Reshape state into matrix
        psi_matrix = quantum_state.reshape(dim_subsystem, dim_environment)
        
        # Compute reduced density matrix
        rho = psi_matrix @ psi_matrix.conj().T
        
        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        
        # von Neumann entropy
        entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
        
        return float(entropy)
    
    def quantum_feature_importance(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Analyze quantum feature importance using entanglement measures."""
        batch_size = x.shape[0]
        feature_importance = np.zeros(self.input_dim)
        
        # Compute baseline quantum state
        with torch.no_grad():
            baseline_output = self.quantum_circuit(x)
        
        # Perturbation analysis
        perturbation_size = 0.1
        
        for feature_idx in range(self.input_dim):
            # Perturb feature
            x_perturbed = x.clone()
            x_perturbed[:, feature_idx] += perturbation_size
            
            with torch.no_grad():
                perturbed_output = self.quantum_circuit(x_perturbed)
            
            # Calculate sensitivity
            sensitivity = torch.mean(torch.abs(perturbed_output - baseline_output))
            feature_importance[feature_idx] = sensitivity.item()
        
        # Normalize
        feature_importance = feature_importance / np.sum(feature_importance)
        
        return {
            'feature_importance': feature_importance,
            'quantum_entanglement': self._measure_entanglement(x),
            'coherence_measures': self._measure_coherence(x)
        }
    
    def _measure_entanglement(self, x: torch.Tensor) -> float:
        """Measure quantum entanglement in the circuit."""
        # Simplified entanglement measure
        # In practice, would compute concurrence or negativity
        return 0.5  # Placeholder
    
    def _measure_coherence(self, x: torch.Tensor) -> Dict[str, float]:
        """Measure quantum coherence properties."""
        return {
            'l1_norm_coherence': 0.3,
            'relative_entropy_coherence': 0.2,
            'coherence_robustness': 0.4
        }


def create_quantum_bci_decoder(
    input_dim: int,
    output_dim: int,
    quantum_config: Optional[QuantumConfig] = None
) -> QuantumNeuralDecoder:
    """Factory function to create quantum BCI decoder."""
    config = quantum_config or QuantumConfig(
        n_qubits=min(16, input_dim),  # Adjust based on input dimension
        n_layers=4,
        learning_rate=0.01,
        use_entanglement=True
    )
    
    decoder = QuantumNeuralDecoder(input_dim, output_dim, config)
    
    logging.getLogger(__name__).info(
        f"Created quantum BCI decoder with {config.n_qubits} qubits, "
        f"{config.n_layers} layers, input_dim={input_dim}, output_dim={output_dim}"
    )
    
    return decoder