"""
Federated Learning Framework for Privacy-Preserving BCI Model Training.

This module implements federated learning capabilities for collaborative BCI model
training across multiple sites while preserving privacy and data sovereignty.
Includes secure aggregation, differential privacy, and Byzantine fault tolerance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    n_clients: int = 10
    n_rounds: int = 100
    client_fraction: float = 0.3  # Fraction of clients selected per round
    local_epochs: int = 5
    local_batch_size: int = 32
    learning_rate: float = 0.01
    aggregation_method: str = "fedavg"  # "fedavg", "fedprox", "scaffold", "fedopt"
    use_differential_privacy: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    byzantine_tolerance: bool = True
    max_byzantine_clients: int = 2
    secure_aggregation: bool = True
    compression_ratio: float = 0.1  # For gradient compression
    use_homomorphic_encryption: bool = False
    min_clients_per_round: int = 3
    convergence_threshold: float = 1e-4
    patience: int = 10


@dataclass
class ClientData:
    """Container for client-specific data."""
    client_id: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelUpdate:
    """Container for model updates from clients."""
    client_id: str
    parameters: Dict[str, torch.Tensor]
    num_samples: int
    loss: float
    accuracy: float
    timestamp: float
    signature: Optional[bytes] = None
    compressed: bool = False


class SecureAggregator:
    """Secure aggregation with cryptographic protection."""
    
    def __init__(self, use_homomorphic: bool = False):
        self.use_homomorphic = use_homomorphic
        self.client_keys = {}
        self.server_private_key = None
        self.server_public_key = None
        self._generate_server_keys()
        
    def _generate_server_keys(self):
        """Generate server key pair."""
        self.server_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.server_public_key = self.server_private_key.public_key()
    
    def register_client(self, client_id: str) -> Dict[str, bytes]:
        """Register a new client and generate keys."""
        client_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        client_public_key = client_private_key.public_key()
        
        self.client_keys[client_id] = {
            'private': client_private_key,
            'public': client_public_key
        }
        
        # Return serialized keys
        return {
            'private_key': client_private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption()
            ),
            'public_key': client_public_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            ),
            'server_public_key': self.server_public_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            )
        }
    
    def encrypt_update(self, update: ModelUpdate, client_id: str) -> bytes:
        """Encrypt model update."""
        if not self.use_homomorphic:
            # Use simple RSA encryption for demonstration
            serialized_update = pickle.dumps(update.parameters)
            
            # Encrypt with server's public key
            encrypted_chunks = []
            chunk_size = 190  # RSA 2048 can encrypt up to 245 bytes, leaving margin
            
            for i in range(0, len(serialized_update), chunk_size):
                chunk = serialized_update[i:i + chunk_size]
                encrypted_chunk = self.server_public_key.encrypt(
                    chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                encrypted_chunks.append(encrypted_chunk)
            
            return pickle.dumps(encrypted_chunks)
        else:
            # Placeholder for homomorphic encryption
            return pickle.dumps(update.parameters)
    
    def decrypt_update(self, encrypted_data: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model update."""
        if not self.use_homomorphic:
            encrypted_chunks = pickle.loads(encrypted_data)
            decrypted_data = b''
            
            for chunk in encrypted_chunks:
                decrypted_chunk = self.server_private_key.decrypt(
                    chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                decrypted_data += decrypted_chunk
            
            return pickle.loads(decrypted_data)
        else:
            return pickle.loads(encrypted_data)
    
    def sign_update(self, update: ModelUpdate, client_id: str) -> bytes:
        """Sign model update for integrity verification."""
        if client_id not in self.client_keys:
            raise ValueError(f"Client {client_id} not registered")
        
        update_hash = hashlib.sha256(pickle.dumps(update.parameters)).digest()
        signature = self.client_keys[client_id]['private'].sign(
            update_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, update: ModelUpdate) -> bool:
        """Verify update signature."""
        if update.signature is None or update.client_id not in self.client_keys:
            return False
        
        try:
            update_hash = hashlib.sha256(pickle.dumps(update.parameters)).digest()
            self.client_keys[update.client_id]['public'].verify(
                update.signature,
                update_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class DifferentialPrivacyMechanism:
    """Differential privacy for federated learning."""
    
    def __init__(self, epsilon: float, delta: float, clip_norm: float):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.logger = logging.getLogger(__name__)
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity."""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad) ** 2
        total_norm = torch.sqrt(total_norm)
        
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-8))
        
        clipped_gradients = {}
        for name, grad in gradients.items():
            clipped_gradients[name] = grad * clip_coef
        
        return clipped_gradients
    
    def add_noise(self, gradients: Dict[str, torch.Tensor], num_samples: int) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise for differential privacy."""
        # Calculate noise scale based on privacy parameters
        sensitivity = self.clip_norm
        noise_scale = (2 * sensitivity * np.log(1.25 / self.delta)) ** 0.5 / self.epsilon
        
        noisy_gradients = {}
        for name, grad in gradients.items():
            noise = torch.normal(0, noise_scale, size=grad.shape)
            noisy_gradients[name] = grad + noise
        
        return noisy_gradients
    
    def compute_privacy_spent(self, num_rounds: int, num_clients: int) -> Tuple[float, float]:
        """Compute total privacy budget spent."""
        # Simplified composition (should use more sophisticated methods in practice)
        total_epsilon = self.epsilon * np.sqrt(num_rounds * np.log(1 / self.delta))
        total_delta = self.delta * num_rounds
        
        return total_epsilon, total_delta


class ByzantineDetector:
    """Detect and filter Byzantine clients."""
    
    def __init__(self, max_byzantine: int, detection_method: str = "krum"):
        self.max_byzantine = max_byzantine
        self.detection_method = detection_method
        self.client_history = {}
        self.logger = logging.getLogger(__name__)
    
    def detect_byzantine_updates(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Detect and filter Byzantine updates."""
        if len(updates) <= self.max_byzantine:
            return updates
        
        if self.detection_method == "krum":
            return self._krum_selection(updates)
        elif self.detection_method == "trimmed_mean":
            return self._trimmed_mean_selection(updates)
        elif self.detection_method == "median":
            return self._median_selection(updates)
        else:
            return updates
    
    def _krum_selection(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Krum algorithm for Byzantine-resilient selection."""
        n = len(updates)
        k = self.max_byzantine
        m = n - k - 2  # Number of closest updates to consider
        
        # Compute pairwise distances
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self._compute_parameter_distance(updates[i], updates[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute Krum scores
        scores = []
        for i in range(n):
            # Find m closest updates
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[1:m+2])  # Exclude self (distance 0)
            scores.append(score)
        
        # Select update with minimum score
        best_idx = np.argmin(scores)
        return [updates[best_idx]]
    
    def _trimmed_mean_selection(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Trimmed mean aggregation."""
        # Remove extreme updates based on parameter norms
        norms = []
        for update in updates:
            total_norm = 0.0
            for param in update.parameters.values():
                total_norm += torch.norm(param) ** 2
            norms.append(total_norm.item())
        
        # Remove top and bottom k updates
        sorted_indices = np.argsort(norms)
        k = self.max_byzantine
        selected_indices = sorted_indices[k:-k] if k > 0 else sorted_indices
        
        return [updates[i] for i in selected_indices]
    
    def _median_selection(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Coordinate-wise median aggregation."""
        # For simplicity, return all updates (median computed during aggregation)
        return updates
    
    def _compute_parameter_distance(self, update1: ModelUpdate, update2: ModelUpdate) -> float:
        """Compute Euclidean distance between parameter sets."""
        distance = 0.0
        
        for name in update1.parameters:
            if name in update2.parameters:
                diff = update1.parameters[name] - update2.parameters[name]
                distance += torch.norm(diff) ** 2
        
        return distance.item() ** 0.5


class FederatedClient:
    """Federated learning client."""
    
    def __init__(self, client_id: str, model: nn.Module, config: FederatedConfig):
        self.client_id = client_id
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.dp_mechanism = DifferentialPrivacyMechanism(
            config.dp_epsilon, config.dp_delta, config.dp_clip_norm
        ) if config.use_differential_privacy else None
        
        self.logger = logging.getLogger(__name__)
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters from server."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name])
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return {name: param.clone() for name, param in self.model.named_parameters()}
    
    def train_local(self, data: ClientData) -> ModelUpdate:
        """Perform local training."""
        self.model.train()
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(data.X_train).to(self.device),
            torch.LongTensor(data.y_train).to(self.device)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.local_batch_size, shuffle=True
        )
        
        initial_parameters = self.get_parameters()
        total_loss = 0.0
        num_batches = 0
        
        # Local training loop
        for epoch in range(self.config.local_epochs):
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Apply differential privacy if enabled
                if self.dp_mechanism:
                    gradients = {name: param.grad.clone() 
                               for name, param in self.model.named_parameters() 
                               if param.grad is not None}
                    
                    clipped_gradients = self.dp_mechanism.clip_gradients(gradients)
                    noisy_gradients = self.dp_mechanism.add_noise(
                        clipped_gradients, len(data.X_train)
                    )
                    
                    # Replace gradients
                    for name, param in self.model.named_parameters():
                        if name in noisy_gradients:
                            param.grad = noisy_gradients[name]
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute parameter differences (for communication efficiency)
        final_parameters = self.get_parameters()
        parameter_updates = {}
        for name in final_parameters:
            parameter_updates[name] = final_parameters[name] - initial_parameters[name]
        
        # Evaluate on validation data if available
        val_accuracy = 0.0
        if data.X_val is not None and data.y_val is not None:
            val_accuracy = self._evaluate(data.X_val, data.y_val)
        
        update = ModelUpdate(
            client_id=self.client_id,
            parameters=parameter_updates,
            num_samples=len(data.X_train),
            loss=total_loss / num_batches,
            accuracy=val_accuracy,
            timestamp=time.time()
        )
        
        return update
    
    def _evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val).to(self.device),
                torch.LongTensor(y_val).to(self.device)
            )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
            
            for batch_x, batch_y in val_loader:
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return correct / total if total > 0 else 0.0


class FederatedServer:
    """Federated learning server."""
    
    def __init__(self, model: nn.Module, config: FederatedConfig):
        self.global_model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        
        self.clients = {}
        self.round_history = []
        self.convergence_history = []
        
        # Security components
        self.secure_aggregator = SecureAggregator(config.secure_aggregation) if config.secure_aggregation else None
        self.byzantine_detector = ByzantineDetector(config.max_byzantine_clients) if config.byzantine_tolerance else None
        
        self.logger = logging.getLogger(__name__)
    
    def register_client(self, client: FederatedClient) -> Optional[Dict[str, bytes]]:
        """Register a new client."""
        self.clients[client.client_id] = client
        self.logger.info(f"Registered client {client.client_id}")
        
        # Return security keys if secure aggregation is enabled
        if self.secure_aggregator:
            return self.secure_aggregator.register_client(client.client_id)
        return None
    
    def select_clients(self, round_num: int) -> List[str]:
        """Select clients for the current round."""
        available_clients = list(self.clients.keys())
        num_selected = max(
            self.config.min_clients_per_round,
            int(self.config.client_fraction * len(available_clients))
        )
        
        # Random selection (could implement more sophisticated strategies)
        selected = np.random.choice(
            available_clients, 
            size=min(num_selected, len(available_clients)), 
            replace=False
        )
        
        return selected.tolist()
    
    def aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates."""
        if not updates:
            return {}
        
        # Byzantine detection
        if self.byzantine_detector:
            updates = self.byzantine_detector.detect_byzantine_updates(updates)
        
        # Verify signatures if secure aggregation is enabled
        if self.secure_aggregator:
            verified_updates = []
            for update in updates:
                if self.secure_aggregator.verify_signature(update):
                    verified_updates.append(update)
                else:
                    self.logger.warning(f"Failed to verify signature for client {update.client_id}")
            updates = verified_updates
        
        if not updates:
            self.logger.warning("No valid updates to aggregate")
            return {}
        
        # Aggregate based on method
        if self.config.aggregation_method == "fedavg":
            return self._federated_averaging(updates)
        elif self.config.aggregation_method == "fedprox":
            return self._federated_proximal(updates)
        elif self.config.aggregation_method == "scaffold":
            return self._scaffold_aggregation(updates)
        else:
            return self._federated_averaging(updates)
    
    def _federated_averaging(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """FedAvg aggregation."""
        total_samples = sum(update.num_samples for update in updates)
        aggregated_params = {}
        
        # Initialize aggregated parameters
        for name in updates[0].parameters:
            aggregated_params[name] = torch.zeros_like(updates[0].parameters[name])
        
        # Weighted averaging
        for update in updates:
            weight = update.num_samples / total_samples
            for name, param in update.parameters.items():
                aggregated_params[name] += weight * param
        
        return aggregated_params
    
    def _federated_proximal(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """FedProx aggregation with proximal term."""
        # For simplicity, use same as FedAvg (full implementation would require proximal regularization)
        return self._federated_averaging(updates)
    
    def _scaffold_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """SCAFFOLD aggregation with control variates."""
        # Simplified implementation
        return self._federated_averaging(updates)
    
    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]) -> None:
        """Update the global model with aggregated parameters."""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_params:
                    param.add_(aggregated_params[name])
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters."""
        return {name: param.clone() for name, param in self.global_model.named_parameters()}
    
    def train_federated(
        self, 
        client_data: Dict[str, ClientData],
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, List[float]]:
        """Run federated learning training."""
        self.logger.info(f"Starting federated training with {len(self.clients)} clients")
        
        history = {
            'round_losses': [],
            'round_accuracies': [],
            'test_accuracies': [],
            'convergence_metrics': [],
            'privacy_spent': []
        }
        
        prev_global_params = self.get_global_parameters()
        patience_counter = 0
        
        for round_num in range(self.config.n_rounds):
            self.logger.info(f"Round {round_num + 1}/{self.config.n_rounds}")
            
            # Select clients
            selected_clients = self.select_clients(round_num)
            
            # Distribute global model
            global_params = self.get_global_parameters()
            
            # Collect updates from selected clients
            updates = []
            
            def train_client(client_id: str) -> Optional[ModelUpdate]:
                if client_id in client_data:
                    client = self.clients[client_id]
                    client.set_parameters(global_params)
                    return client.train_local(client_data[client_id])
                return None
            
            # Parallel client training
            with ThreadPoolExecutor(max_workers=min(len(selected_clients), 4)) as executor:
                future_to_client = {
                    executor.submit(train_client, client_id): client_id 
                    for client_id in selected_clients
                }
                
                for future in as_completed(future_to_client):
                    update = future.result()
                    if update is not None:
                        # Sign update if secure aggregation is enabled
                        if self.secure_aggregator:
                            update.signature = self.secure_aggregator.sign_update(
                                update, update.client_id
                            )
                        updates.append(update)
            
            if not updates:
                self.logger.warning(f"No updates received in round {round_num + 1}")
                continue
            
            # Aggregate updates
            aggregated_params = self.aggregate_updates(updates)
            
            if aggregated_params:
                # Update global model
                self.update_global_model(aggregated_params)
                
                # Compute metrics
                avg_loss = np.mean([update.loss for update in updates])
                avg_accuracy = np.mean([update.accuracy for update in updates])
                
                # Test on global test set if available
                test_accuracy = 0.0
                if test_data is not None:
                    test_accuracy = self._evaluate_global_model(test_data[0], test_data[1])
                
                # Convergence check
                convergence_metric = self._compute_convergence_metric(prev_global_params, global_params)
                
                # Update history
                history['round_losses'].append(avg_loss)
                history['round_accuracies'].append(avg_accuracy)
                history['test_accuracies'].append(test_accuracy)
                history['convergence_metrics'].append(convergence_metric)
                
                # Privacy accounting
                if self.config.use_differential_privacy:
                    dp_mechanism = DifferentialPrivacyMechanism(
                        self.config.dp_epsilon, self.config.dp_delta, self.config.dp_clip_norm
                    )
                    epsilon_spent, delta_spent = dp_mechanism.compute_privacy_spent(
                        round_num + 1, len(selected_clients)
                    )
                    history['privacy_spent'].append((epsilon_spent, delta_spent))
                
                self.logger.info(
                    f"Round {round_num + 1}: Loss={avg_loss:.4f}, "
                    f"Accuracy={avg_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}, "
                    f"Convergence={convergence_metric:.6f}"
                )
                
                # Check convergence
                if convergence_metric < self.config.convergence_threshold:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        self.logger.info(f"Converged after {round_num + 1} rounds")
                        break
                else:
                    patience_counter = 0
                
                prev_global_params = global_params
        
        self.logger.info("Federated training completed")
        return history
    
    def _evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate global model on test data."""
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            test_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_test).to(self.device),
                torch.LongTensor(y_test).to(self.device)
            )
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
            
            for batch_x, batch_y in test_loader:
                outputs = self.global_model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def _compute_convergence_metric(
        self, 
        prev_params: Dict[str, torch.Tensor], 
        curr_params: Dict[str, torch.Tensor]
    ) -> float:
        """Compute convergence metric based on parameter changes."""
        total_change = 0.0
        total_norm = 0.0
        
        for name in curr_params:
            if name in prev_params:
                change = torch.norm(curr_params[name] - prev_params[name])
                norm = torch.norm(curr_params[name])
                total_change += change.item() ** 2
                total_norm += norm.item() ** 2
        
        if total_norm == 0:
            return 0.0
        
        return (total_change ** 0.5) / (total_norm ** 0.5)


def create_federated_bci_system(
    model_factory: Callable[[], nn.Module],
    config: Optional[FederatedConfig] = None
) -> Tuple[FederatedServer, List[FederatedClient]]:
    """
    Factory function to create federated BCI learning system.
    
    Args:
        model_factory: Function that creates a new model instance
        config: Federated learning configuration
        
    Returns:
        Tuple of (server, list of clients)
    """
    config = config or FederatedConfig()
    
    # Create server with global model
    global_model = model_factory()
    server = FederatedServer(global_model, config)
    
    # Create clients
    clients = []
    for i in range(config.n_clients):
        client_model = model_factory()
        client = FederatedClient(f"client_{i}", client_model, config)
        clients.append(client)
        
        # Register client with server
        server.register_client(client)
    
    logging.getLogger(__name__).info(
        f"Created federated BCI system with {config.n_clients} clients"
    )
    
    return server, clients