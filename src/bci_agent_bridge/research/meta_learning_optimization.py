"""
Meta-Learning Optimization for Few-Shot BCI Adaptation.

This module implements cutting-edge meta-learning algorithms for rapid BCI
adaptation to new users with minimal calibration data. Features gradient-based
meta-learning (MAML), metric-based approaches (ProtoNet), and neural architecture
search for personalized BCI decoders.

Research Contributions:
- Novel meta-learning framework for BCI user adaptation
- Few-shot learning with <10 calibration trials
- Personalized neural architecture optimization
- Cross-subject transfer learning with domain adaptation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy
import math
from collections import OrderedDict
import random
from scipy.stats import entropy
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import time

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning optimization."""
    
    # Core meta-learning settings
    meta_learning_rate: float = 0.001
    adaptation_learning_rate: float = 0.01
    meta_batch_size: int = 4  # Number of tasks per meta-batch
    adaptation_steps: int = 5  # Inner loop steps
    meta_epochs: int = 1000
    
    # Few-shot learning settings
    n_way: int = 2  # Number of classes (e.g., binary BCI tasks)
    k_shot: int = 5  # Number of support examples per class
    n_query: int = 10  # Number of query examples per class
    
    # Architecture settings
    base_hidden_dim: int = 128
    meta_hidden_dim: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.1
    
    # Optimization settings
    use_first_order: bool = False  # First-order MAML approximation
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    
    # Advanced features
    use_neural_architecture_search: bool = True
    use_domain_adaptation: bool = True
    use_uncertainty_weighting: bool = True
    
    # Convergence criteria
    convergence_threshold: float = 1e-5
    patience: int = 50


class MetaModule(nn.Module):
    """Base class for meta-learnable modules."""
    
    def __init__(self):
        super().__init__()
        self.meta_parameters = []
    
    def meta_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Get meta-learnable parameters as dictionary."""
        return OrderedDict(self.named_parameters())
    
    def fast_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with given parameters (for inner loop)."""
        raise NotImplementedError


class MetaLinear(MetaModule):
    """Meta-learnable linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return F.linear(x, self.weight, self.bias)
    
    def fast_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with given parameters."""
        weight = params.get(f'{id(self)}.weight', self.weight)
        bias = params.get(f'{id(self)}.bias', self.bias) if self.use_bias else None
        return F.linear(x, weight, bias)


class MetaBatchNorm1d(MetaModule):
    """Meta-learnable batch normalization."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (not meta-learnable)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        if self.training:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, True, self.momentum, self.eps
            )
        else:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps
            )
    
    def fast_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with given parameters."""
        weight = params.get(f'{id(self)}.weight', self.weight)
        bias = params.get(f'{id(self)}.bias', self.bias)
        
        # Use batch statistics for fast adaptation
        batch_mean = x.mean(dim=0, keepdim=True)
        batch_var = x.var(dim=0, keepdim=True)
        
        return F.batch_norm(
            x, batch_mean.squeeze(), batch_var.squeeze(),
            weight, bias, True, 0.0, self.eps
        )


class MetaNeuralDecoder(MetaModule):
    """Meta-learnable neural decoder for BCI applications."""
    
    def __init__(self, config: MetaLearningConfig, input_dim: int, output_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build architecture
        layers = []
        prev_dim = input_dim
        
        for i in range(config.num_layers):
            # Linear layer
            layers.append((f'linear_{i}', MetaLinear(prev_dim, config.base_hidden_dim)))
            
            # Batch normalization
            layers.append((f'bn_{i}', MetaBatchNorm1d(config.base_hidden_dim)))
            
            # Activation
            layers.append((f'relu_{i}', nn.ReLU()))
            
            # Dropout
            if config.dropout_rate > 0:
                layers.append((f'dropout_{i}', nn.Dropout(config.dropout_rate)))
            
            prev_dim = config.base_hidden_dim
        
        # Output layer
        layers.append(('output', MetaLinear(prev_dim, output_dim)))
        
        self.layers = nn.ModuleDict(OrderedDict(layers))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        for name, layer in self.layers.items():
            if isinstance(layer, (MetaLinear, MetaBatchNorm1d)):
                x = layer(x)
            else:
                x = layer(x)
        return x
    
    def fast_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with given parameters (for inner loop)."""
        for name, layer in self.layers.items():
            if isinstance(layer, (MetaLinear, MetaBatchNorm1d)):
                x = layer.fast_forward(x, params)
            else:
                x = layer(x)
        return x
    
    def get_param_dict(self) -> Dict[str, torch.Tensor]:
        """Get parameters as dictionary with layer IDs."""
        param_dict = {}
        for name, layer in self.layers.items():
            if isinstance(layer, (MetaLinear, MetaBatchNorm1d)):
                for param_name, param in layer.named_parameters():
                    param_dict[f'{id(layer)}.{param_name}'] = param
        return param_dict


class MAML:
    """Model-Agnostic Meta-Learning for BCI adaptation."""
    
    def __init__(self, model: MetaNeuralDecoder, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.meta_learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(__name__)
    
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        adaptation_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Adapt model to new task using support data."""
        adaptation_steps = adaptation_steps or self.config.adaptation_steps
        
        # Clone model parameters for adaptation
        adapted_params = self.model.get_param_dict()
        adapted_params = {k: v.clone().requires_grad_(True) for k, v in adapted_params.items()}
        
        # Inner loop: adapt to support data
        for step in range(adaptation_steps):
            # Forward pass with current adapted parameters
            logits = self.model.fast_forward(support_x, adapted_params)
            loss = self.criterion(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.config.use_first_order,
                retain_graph=True
            )
            
            # Update adapted parameters
            adapted_params = {
                name: param - self.config.adaptation_learning_rate * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
            }
        
        return adapted_params
    
    def meta_train_step(self, batch_tasks: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Perform one meta-training step."""
        meta_losses = []
        meta_accuracies = []
        
        self.meta_optimizer.zero_grad()
        
        for task_data in batch_tasks:
            support_x = task_data['support_x'].to(self.device)
            support_y = task_data['support_y'].to(self.device)
            query_x = task_data['query_x'].to(self.device)
            query_y = task_data['query_y'].to(self.device)
            
            # Fast adaptation on support set
            adapted_params = self.adapt(support_x, support_y)
            
            # Evaluate on query set
            query_logits = self.model.fast_forward(query_x, adapted_params)
            query_loss = self.criterion(query_logits, query_y)
            
            # Accumulate meta-loss
            meta_losses.append(query_loss)
            
            # Calculate accuracy
            with torch.no_grad():
                pred = torch.argmax(query_logits, dim=1)
                accuracy = (pred == query_y).float().mean()
                meta_accuracies.append(accuracy.item())
        
        # Meta-gradient step
        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
        
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'meta_accuracy': np.mean(meta_accuracies)
        }
    
    def meta_train(self, meta_dataset: List[Dict[str, torch.Tensor]]) -> Dict[str, List[float]]:
        """Complete meta-training process."""
        history = {
            'meta_losses': [],
            'meta_accuracies': [],
            'adaptation_times': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.meta_epochs):
            epoch_start_time = time.time()
            
            # Sample meta-batch
            batch_tasks = random.sample(meta_dataset, self.config.meta_batch_size)
            
            # Meta-training step
            metrics = self.meta_train_step(batch_tasks)
            
            # Update history
            history['meta_losses'].append(metrics['meta_loss'])
            history['meta_accuracies'].append(metrics['meta_accuracy'])
            history['adaptation_times'].append(time.time() - epoch_start_time)
            
            # Early stopping
            if metrics['meta_loss'] < best_loss - self.config.convergence_threshold:
                best_loss = metrics['meta_loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                self.logger.info(f"Meta-training converged after {epoch + 1} epochs")
                break
            
            # Logging
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}: Meta-loss={metrics['meta_loss']:.4f}, "
                    f"Meta-accuracy={metrics['meta_accuracy']:.4f}"
                )
        
        return history


class ProtoNet:
    """Prototypical Networks for few-shot BCI classification."""
    
    def __init__(self, embedding_net: nn.Module, config: MetaLearningConfig):
        self.embedding_net = embedding_net
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_net.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.embedding_net.parameters(),
            lr=config.meta_learning_rate
        )
        
    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute class prototypes from support embeddings."""
        n_classes = len(torch.unique(support_labels))
        embedding_dim = support_embeddings.size(1)
        
        prototypes = torch.zeros(n_classes, embedding_dim).to(self.device)
        
        for class_idx in range(n_classes):
            class_mask = (support_labels == class_idx)
            if class_mask.sum() > 0:
                prototypes[class_idx] = support_embeddings[class_mask].mean(dim=0)
        
        return prototypes
    
    def classify_queries(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Classify queries based on distance to prototypes."""
        # Compute distances to all prototypes
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        
        # Convert to logits (negative distances)
        logits = -distances
        
        return logits
    
    def train_step(self, batch_tasks: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Training step for prototypical networks."""
        total_loss = 0.0
        total_accuracy = 0.0
        
        self.optimizer.zero_grad()
        
        for task_data in batch_tasks:
            support_x = task_data['support_x'].to(self.device)
            support_y = task_data['support_y'].to(self.device)
            query_x = task_data['query_x'].to(self.device)
            query_y = task_data['query_y'].to(self.device)
            
            # Embed support and query examples
            support_embeddings = self.embedding_net(support_x)
            query_embeddings = self.embedding_net(query_x)
            
            # Compute prototypes
            prototypes = self.compute_prototypes(support_embeddings, support_y)
            
            # Classify queries
            logits = self.classify_queries(query_embeddings, prototypes)
            
            # Compute loss
            loss = F.cross_entropy(logits, query_y)
            total_loss += loss
            
            # Compute accuracy
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                accuracy = (pred == query_y).float().mean()
                total_accuracy += accuracy.item()
        
        # Backward pass
        avg_loss = total_loss / len(batch_tasks)
        avg_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': avg_loss.item(),
            'accuracy': total_accuracy / len(batch_tasks)
        }


class NeuralArchitectureSearch:
    """Neural Architecture Search for personalized BCI decoders."""
    
    def __init__(self, config: MetaLearningConfig, input_dim: int, output_dim: int):
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Architecture search space
        self.search_space = {
            'num_layers': [2, 3, 4, 5],
            'hidden_dims': [64, 128, 256, 512],
            'activation_functions': ['relu', 'gelu', 'elu', 'swish'],
            'normalization': ['batch_norm', 'layer_norm', 'none'],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3]
        }
        
        self.logger = logging.getLogger(__name__)
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample random architecture from search space."""
        return {
            'num_layers': random.choice(self.search_space['num_layers']),
            'hidden_dim': random.choice(self.search_space['hidden_dims']),
            'activation': random.choice(self.search_space['activation_functions']),
            'normalization': random.choice(self.search_space['normalization']),
            'dropout_rate': random.choice(self.search_space['dropout_rates'])
        }
    
    def build_model(self, arch_config: Dict[str, Any]) -> nn.Module:
        """Build model from architecture configuration."""
        layers = []
        prev_dim = self.input_dim
        
        for i in range(arch_config['num_layers']):
            # Linear layer
            layers.append(nn.Linear(prev_dim, arch_config['hidden_dim']))
            
            # Normalization
            if arch_config['normalization'] == 'batch_norm':
                layers.append(nn.BatchNorm1d(arch_config['hidden_dim']))
            elif arch_config['normalization'] == 'layer_norm':
                layers.append(nn.LayerNorm(arch_config['hidden_dim']))
            
            # Activation
            if arch_config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif arch_config['activation'] == 'gelu':
                layers.append(nn.GELU())
            elif arch_config['activation'] == 'elu':
                layers.append(nn.ELU())
            elif arch_config['activation'] == 'swish':
                layers.append(nn.SiLU())  # SiLU is the same as Swish
            
            # Dropout
            if arch_config['dropout_rate'] > 0:
                layers.append(nn.Dropout(arch_config['dropout_rate']))
            
            prev_dim = arch_config['hidden_dim']
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def evaluate_architecture(
        self,
        arch_config: Dict[str, Any],
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        num_epochs: int = 50
    ) -> float:
        """Evaluate architecture performance."""
        model = self.build_model(arch_config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        train_x, train_y = train_data
        val_x, val_y = val_data
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(train_x)
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_pred == val_y).float().mean().item()
        
        return val_accuracy
    
    def search_best_architecture(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        num_trials: int = 50
    ) -> Tuple[Dict[str, Any], float]:
        """Search for best architecture using random search."""
        best_arch = None
        best_performance = 0.0
        
        self.logger.info(f"Starting architecture search with {num_trials} trials")
        
        for trial in range(num_trials):
            arch_config = self.sample_architecture()
            performance = self.evaluate_architecture(arch_config, train_data, val_data)
            
            if performance > best_performance:
                best_performance = performance
                best_arch = arch_config
                self.logger.info(f"Trial {trial}: New best architecture with {performance:.4f} accuracy")
        
        return best_arch, best_performance


class DomainAdaptation:
    """Domain adaptation for cross-subject BCI transfer."""
    
    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Domain discriminator for adversarial training
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_extractor[-2].out_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Source vs Target domain
        ).to(self.device)
        
        self.feature_optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=0.001)
        self.classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        self.discriminator_optimizer = torch.optim.Adam(self.domain_discriminator.parameters(), lr=0.001)
        
    def dann_loss(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        target_x: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Domain Adversarial Neural Network loss."""
        
        # Extract features
        source_features = self.feature_extractor(source_x)
        target_features = self.feature_extractor(target_x)
        
        # Classification loss on source domain
        source_pred = self.classifier(source_features)
        class_loss = F.cross_entropy(source_pred, source_y)
        
        # Domain classification
        all_features = torch.cat([source_features, target_features], dim=0)
        domain_pred = self.domain_discriminator(all_features)
        
        # Domain labels (0 for source, 1 for target)
        domain_labels = torch.cat([
            torch.zeros(source_x.size(0), dtype=torch.long),
            torch.ones(target_x.size(0), dtype=torch.long)
        ]).to(self.device)
        
        domain_loss = F.cross_entropy(domain_pred, domain_labels)
        
        # Total loss with gradient reversal for feature extractor
        total_loss = class_loss - alpha * domain_loss
        
        return total_loss, {
            'class_loss': class_loss.item(),
            'domain_loss': domain_loss.item(),
            'total_loss': total_loss.item()
        }


class UncertaintyAwareMetaLearning:
    """Meta-learning with uncertainty quantification."""
    
    def __init__(self, base_model: MetaNeuralDecoder, config: MetaLearningConfig):
        self.base_model = base_model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(config.base_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Log variance
            nn.Softplus()  # Ensure positive variance
        ).to(self.device)
    
    def forward_with_uncertainty(
        self,
        x: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        # Get features from second-to-last layer
        features = x
        layers = list(self.base_model.layers.items())[:-1]  # Exclude output layer
        
        for name, layer in layers:
            if isinstance(layer, (MetaLinear, MetaBatchNorm1d)) and params is not None:
                features = layer.fast_forward(features, params)
            elif isinstance(layer, (MetaLinear, MetaBatchNorm1d)):
                features = layer(features)
            else:
                features = layer(features)
        
        # Main prediction
        if params is not None:
            output_layer = self.base_model.layers['output']
            logits = output_layer.fast_forward(features, params)
        else:
            logits = self.base_model.layers['output'](features)
        
        # Uncertainty estimation
        log_variance = self.uncertainty_net(features)
        
        return logits, log_variance
    
    def uncertainty_weighted_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        log_variance: torch.Tensor
    ) -> torch.Tensor:
        """Compute uncertainty-weighted loss."""
        # Aleatoric uncertainty weighting
        precision = torch.exp(-log_variance)
        loss = precision * F.cross_entropy(predictions, targets, reduction='none')
        loss = loss + 0.5 * log_variance.squeeze()
        return loss.mean()


def create_few_shot_bci_system(
    input_dim: int,
    output_dim: int,
    config: Optional[MetaLearningConfig] = None
) -> Dict[str, Any]:
    """
    Create complete few-shot BCI meta-learning system.
    
    Returns:
        Dictionary containing all meta-learning components
    """
    config = config or MetaLearningConfig()
    
    # Core meta-learning model
    meta_model = MetaNeuralDecoder(config, input_dim, output_dim)
    
    # MAML trainer
    maml = MAML(meta_model, config)
    
    # Prototypical networks (alternative approach)
    embedding_net = nn.Sequential(
        nn.Linear(input_dim, config.meta_hidden_dim),
        nn.ReLU(),
        nn.Linear(config.meta_hidden_dim, config.meta_hidden_dim),
        nn.ReLU(),
        nn.Linear(config.meta_hidden_dim, 128)  # Embedding dimension
    )
    protonet = ProtoNet(embedding_net, config)
    
    # Neural Architecture Search
    nas = NeuralArchitectureSearch(config, input_dim, output_dim)
    
    # Uncertainty-aware learning
    uncertainty_learner = UncertaintyAwareMetaLearning(meta_model, config)
    
    logger.info(f"Created few-shot BCI system with {input_dim}D input, {output_dim}D output")
    
    return {
        'meta_model': meta_model,
        'maml': maml,
        'protonet': protonet,
        'nas': nas,
        'uncertainty_learner': uncertainty_learner,
        'config': config
    }


def generate_synthetic_bci_tasks(
    num_tasks: int,
    input_dim: int,
    config: MetaLearningConfig
) -> List[Dict[str, torch.Tensor]]:
    """Generate synthetic BCI tasks for meta-learning evaluation."""
    tasks = []
    
    for task_id in range(num_tasks):
        # Generate synthetic neural patterns for this "subject"
        pattern_mean = np.random.randn(input_dim) * 0.5
        pattern_std = 0.1 + np.random.rand() * 0.2
        
        # Support set
        support_x = []
        support_y = []
        
        for class_id in range(config.n_way):
            class_pattern = pattern_mean + np.random.randn(input_dim) * 0.3 * class_id
            
            for shot in range(config.k_shot):
                sample = class_pattern + np.random.randn(input_dim) * pattern_std
                support_x.append(sample)
                support_y.append(class_id)
        
        # Query set
        query_x = []
        query_y = []
        
        for class_id in range(config.n_way):
            class_pattern = pattern_mean + np.random.randn(input_dim) * 0.3 * class_id
            
            for query in range(config.n_query):
                sample = class_pattern + np.random.randn(input_dim) * pattern_std
                query_x.append(sample)
                query_y.append(class_id)
        
        # Convert to tensors
        task = {
            'support_x': torch.FloatTensor(support_x),
            'support_y': torch.LongTensor(support_y),
            'query_x': torch.FloatTensor(query_x),
            'query_y': torch.LongTensor(query_y),
            'task_id': task_id
        }
        
        tasks.append(task)
    
    return tasks


# Evaluation metrics for meta-learning
def evaluate_few_shot_performance(
    meta_learner: MAML,
    test_tasks: List[Dict[str, torch.Tensor]],
    num_adaptation_steps: List[int] = [1, 5, 10]
) -> Dict[str, List[float]]:
    """Evaluate few-shot learning performance."""
    results = {f'{k}_shot_accuracy': [] for k in num_adaptation_steps}
    
    meta_learner.model.eval()
    
    for task in test_tasks:
        support_x = task['support_x'].to(meta_learner.device)
        support_y = task['support_y'].to(meta_learner.device)
        query_x = task['query_x'].to(meta_learner.device)
        query_y = task['query_y'].to(meta_learner.device)
        
        for k in num_adaptation_steps:
            # Adapt to task
            adapted_params = meta_learner.adapt(support_x, support_y, k)
            
            # Test on query set
            with torch.no_grad():
                query_logits = meta_learner.model.fast_forward(query_x, adapted_params)
                pred = torch.argmax(query_logits, dim=1)
                accuracy = (pred == query_y).float().mean().item()
                results[f'{k}_shot_accuracy'].append(accuracy)
    
    # Average results
    return {k: np.mean(v) for k, v in results.items()}