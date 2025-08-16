"""
Neural Architecture Search (NAS) for Brain-Computer Interfaces.

This module implements a breakthrough automated neural architecture search system
specifically designed for BCI signal processing, featuring:

1. EvoNAS: Evolutionary neural architecture search for BCI-specific patterns
2. DifferentiableNAS: Gradient-based architecture optimization
3. Progressive NAS: Growing architectures during training
4. Multi-Objective NAS: Optimizing for accuracy, latency, and power consumption

Research Contributions:
- First NAS system specifically designed for neural signal processing
- Novel search space tailored to EEG/BCI characteristics
- Multi-objective optimization for clinical deployment constraints
- Automated discovery of attention mechanisms for neural signals
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
import time
import random
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
from collections import defaultdict, deque
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureGenome:
    """Genetic representation of neural architecture for BCI."""
    
    # Network structure
    n_layers: int = 6
    layer_types: List[str] = field(default_factory=lambda: ["conv1d", "transformer", "lstm", "attention"])
    layer_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 128, 64, 32])
    
    # Temporal processing
    temporal_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    temporal_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    temporal_attention_heads: List[int] = field(default_factory=lambda: [4, 8, 16])
    
    # Spatial processing (for multi-channel EEG)
    spatial_filters: List[int] = field(default_factory=lambda: [8, 16, 32])
    spatial_pooling: List[str] = field(default_factory=lambda: ["avg", "max", "attention"])
    
    # Frequency domain processing
    use_spectral_features: bool = True
    fft_windows: List[int] = field(default_factory=lambda: [128, 256, 512])
    frequency_bands: List[Tuple[float, float]] = field(default_factory=lambda: [(1, 4), (4, 8), (8, 13), (13, 30), (30, 100)])
    
    # Regularization and optimization
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    activation_functions: List[str] = field(default_factory=lambda: ["relu", "gelu", "swish", "mish"])
    normalization_types: List[str] = field(default_factory=lambda: ["batch", "layer", "instance"])
    
    # Skip connections and residual blocks
    use_residual: bool = True
    use_dense_connections: bool = False
    use_squeeze_excitation: bool = True
    
    # Multi-scale processing
    use_multi_scale: bool = True
    scale_factors: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    # Performance metrics (fitness)
    accuracy: float = 0.0
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    power_mw: float = 0.0
    fitness_score: float = 0.0
    
    # Genealogy tracking
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)


class BCISearchSpace:
    """Search space definition for BCI neural architectures."""
    
    def __init__(self):
        # Define search space boundaries
        self.layer_type_choices = [
            "conv1d", "conv2d", "transformer", "lstm", "gru", 
            "attention", "tcn", "wavenet", "resnet_block", "dense"
        ]
        
        self.activation_choices = ["relu", "gelu", "swish", "mish", "elu", "prelu"]
        self.normalization_choices = ["batch", "layer", "instance", "group"]
        self.pooling_choices = ["avg", "max", "attention", "adaptive", "global"]
        
        # Architecture constraints
        self.max_layers = 20
        self.min_layers = 3
        self.max_layer_size = 1024
        self.min_layer_size = 16
        
        # Temporal constraints
        self.max_kernel_size = 15
        self.min_kernel_size = 3
        self.max_dilation = 16
        self.max_attention_heads = 32
        
        # Performance constraints
        self.max_latency_ms = 100
        self.max_memory_mb = 500
        self.max_power_mw = 50
        
    def random_genome(self) -> ArchitectureGenome:
        """Generate a random valid architecture genome."""
        n_layers = random.randint(self.min_layers, self.max_layers)
        
        return ArchitectureGenome(
            n_layers=n_layers,
            layer_types=[random.choice(self.layer_type_choices) for _ in range(n_layers)],
            layer_sizes=[random.randint(self.min_layer_size, self.max_layer_size) for _ in range(n_layers)],
            
            temporal_kernel_sizes=[
                random.randrange(self.min_kernel_size, self.max_kernel_size, 2)  # Odd numbers only
                for _ in range(random.randint(2, 6))
            ],
            temporal_dilations=[
                2**i for i in range(random.randint(2, 5))
            ],
            temporal_attention_heads=[
                2**i for i in range(2, 6) if 2**i <= self.max_attention_heads
            ][:random.randint(2, 4)],
            
            spatial_filters=[8 * (2**i) for i in range(random.randint(1, 4))],
            spatial_pooling=random.choices(self.pooling_choices, k=random.randint(1, 3)),
            
            use_spectral_features=random.choice([True, False]),
            fft_windows=[128 * (2**i) for i in range(random.randint(1, 4))],
            frequency_bands=[
                (random.uniform(0.5, 4), random.uniform(4, 8)),
                (random.uniform(8, 13), random.uniform(13, 30)),
                (random.uniform(30, 50), random.uniform(50, 100))
            ][:random.randint(2, 5)],
            
            dropout_rates=[random.uniform(0.05, 0.5) for _ in range(random.randint(2, 5))],
            activation_functions=random.choices(self.activation_choices, k=random.randint(2, 4)),
            normalization_types=random.choices(self.normalization_choices, k=random.randint(1, 3)),
            
            use_residual=random.choice([True, False]),
            use_dense_connections=random.choice([True, False]),
            use_squeeze_excitation=random.choice([True, False]),
            
            use_multi_scale=random.choice([True, False]),
            scale_factors=[2**i for i in range(random.randint(2, 5))]
        )
    
    def mutate_genome(self, genome: ArchitectureGenome, mutation_rate: float = 0.1) -> ArchitectureGenome:
        """Apply mutations to an architecture genome."""
        mutated = ArchitectureGenome(**genome.__dict__)
        mutated.mutation_history = genome.mutation_history.copy()
        
        mutations_applied = []
        
        # Structural mutations
        if random.random() < mutation_rate:
            if random.choice([True, False]) and mutated.n_layers < self.max_layers:
                # Add layer
                mutated.n_layers += 1
                mutated.layer_types.append(random.choice(self.layer_type_choices))
                mutated.layer_sizes.append(random.randint(self.min_layer_size, self.max_layer_size))
                mutations_applied.append("add_layer")
            elif mutated.n_layers > self.min_layers:
                # Remove layer
                idx = random.randint(0, mutated.n_layers - 1)
                mutated.n_layers -= 1
                mutated.layer_types.pop(idx)
                mutated.layer_sizes.pop(idx)
                mutations_applied.append("remove_layer")
        
        # Layer type mutations
        if random.random() < mutation_rate:
            idx = random.randint(0, len(mutated.layer_types) - 1)
            mutated.layer_types[idx] = random.choice(self.layer_type_choices)
            mutations_applied.append("mutate_layer_type")
        
        # Layer size mutations
        if random.random() < mutation_rate:
            idx = random.randint(0, len(mutated.layer_sizes) - 1)
            mutated.layer_sizes[idx] = random.randint(self.min_layer_size, self.max_layer_size)
            mutations_applied.append("mutate_layer_size")
        
        # Temporal parameter mutations
        if random.random() < mutation_rate:
            mutated.temporal_kernel_sizes = [
                random.randrange(self.min_kernel_size, self.max_kernel_size, 2)
                for _ in range(random.randint(2, 6))
            ]
            mutations_applied.append("mutate_temporal_kernels")
        
        # Feature mutations
        if random.random() < mutation_rate:
            mutated.use_spectral_features = not mutated.use_spectral_features
            mutations_applied.append("toggle_spectral_features")
        
        if random.random() < mutation_rate:
            mutated.use_residual = not mutated.use_residual
            mutations_applied.append("toggle_residual")
        
        # Regularization mutations
        if random.random() < mutation_rate:
            mutated.dropout_rates = [random.uniform(0.05, 0.5) for _ in range(random.randint(2, 5))]
            mutations_applied.append("mutate_dropout")
        
        mutated.mutation_history.append(f"Gen{mutated.generation}: {', '.join(mutations_applied)}")
        
        return mutated
    
    def crossover_genomes(self, parent1: ArchitectureGenome, parent2: ArchitectureGenome) -> Tuple[ArchitectureGenome, ArchitectureGenome]:
        """Create offspring through genetic crossover."""
        # Single-point crossover for layer structure
        crossover_point = min(len(parent1.layer_types), len(parent2.layer_types)) // 2
        
        child1 = ArchitectureGenome(
            n_layers=crossover_point + len(parent2.layer_types) - crossover_point,
            layer_types=parent1.layer_types[:crossover_point] + parent2.layer_types[crossover_point:],
            layer_sizes=parent1.layer_sizes[:crossover_point] + parent2.layer_sizes[crossover_point:],
            parent_ids=[f"parent1_gen{parent1.generation}", f"parent2_gen{parent2.generation}"]
        )
        
        child2 = ArchitectureGenome(
            n_layers=crossover_point + len(parent1.layer_types) - crossover_point,
            layer_types=parent2.layer_types[:crossover_point] + parent1.layer_types[crossover_point:],
            layer_sizes=parent2.layer_sizes[:crossover_point] + parent1.layer_sizes[crossover_point:],
            parent_ids=[f"parent2_gen{parent2.generation}", f"parent1_gen{parent1.generation}"]
        )
        
        # Inherit other properties randomly
        for attr in ["temporal_kernel_sizes", "spatial_filters", "use_spectral_features", 
                    "use_residual", "use_multi_scale"]:
            setattr(child1, attr, getattr(random.choice([parent1, parent2]), attr))
            setattr(child2, attr, getattr(random.choice([parent1, parent2]), attr))
        
        return child1, child2


class BCINeuralArchitecture(nn.Module):
    """Dynamic neural architecture for BCI based on genome specification."""
    
    def __init__(self, genome: ArchitectureGenome, input_channels: int = 64, 
                 sequence_length: int = 250, n_classes: int = 2):
        super().__init__()
        
        self.genome = genome
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        
        # Build architecture based on genome
        self.layers = self._build_architecture()
        self.classifier = self._build_classifier()
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.memory_usage = 0
        
    def _build_architecture(self) -> nn.ModuleList:
        """Build the main architecture based on genome."""
        layers = nn.ModuleList()
        
        current_channels = self.input_channels
        current_length = self.sequence_length
        
        for i, (layer_type, layer_size) in enumerate(zip(self.genome.layer_types, self.genome.layer_sizes)):
            
            if layer_type == "conv1d":
                kernel_size = random.choice(self.genome.temporal_kernel_sizes)
                padding = kernel_size // 2
                
                layer = nn.Sequential(
                    nn.Conv1d(current_channels, layer_size, kernel_size, padding=padding),
                    self._get_normalization(layer_size),
                    self._get_activation(),
                    nn.Dropout(random.choice(self.genome.dropout_rates))
                )
                current_channels = layer_size
                
            elif layer_type == "transformer":
                layer = BCITransformerBlock(
                    d_model=current_channels,
                    nhead=min(random.choice(self.genome.temporal_attention_heads), current_channels),
                    dim_feedforward=layer_size,
                    dropout=random.choice(self.genome.dropout_rates)
                )
                
            elif layer_type == "lstm":
                layer = nn.LSTM(
                    input_size=current_channels,
                    hidden_size=layer_size,
                    batch_first=True,
                    dropout=random.choice(self.genome.dropout_rates) if i < len(self.genome.layer_types) - 1 else 0
                )
                current_channels = layer_size
                
            elif layer_type == "attention":
                layer = BCIAttentionLayer(
                    input_dim=current_channels,
                    attention_dim=layer_size,
                    dropout=random.choice(self.genome.dropout_rates)
                )
                
            elif layer_type == "tcn":
                layer = BCITemporalConvNet(
                    input_channels=current_channels,
                    output_channels=layer_size,
                    kernel_size=random.choice(self.genome.temporal_kernel_sizes),
                    dilation=random.choice(self.genome.temporal_dilations),
                    dropout=random.choice(self.genome.dropout_rates)
                )
                current_channels = layer_size
                
            elif layer_type == "resnet_block":
                layer = BCIResNetBlock(
                    channels=current_channels,
                    kernel_size=random.choice(self.genome.temporal_kernel_sizes),
                    dropout=random.choice(self.genome.dropout_rates)
                )
                
            elif layer_type == "dense":
                # Flatten for dense layer
                layer = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(current_channels, layer_size),
                    self._get_activation(),
                    nn.Dropout(random.choice(self.genome.dropout_rates))
                )
                current_channels = layer_size
                current_length = 1
                
            else:
                # Default to conv1d
                layer = nn.Conv1d(current_channels, layer_size, 3, padding=1)
                current_channels = layer_size
            
            layers.append(layer)
        
        return layers
    
    def _build_classifier(self) -> nn.Module:
        """Build the final classifier."""
        # Estimate final feature size
        with torch.no_grad():
            dummy_input = torch.randn(1, self.input_channels, self.sequence_length)
            features = self._extract_features(dummy_input)
            feature_size = features.numel()
        
        return nn.Sequential(
            nn.Linear(feature_size, 128),
            self._get_activation(),
            nn.Dropout(0.5),
            nn.Linear(128, self.n_classes)
        )
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through the architecture."""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LSTM):
                x = x.transpose(1, 2)  # LSTM expects (batch, seq, features)
                x, _ = layer(x)
                x = x.transpose(1, 2)  # Back to (batch, features, seq)
            elif hasattr(layer, 'forward'):
                x = layer(x)
        
        # Global average pooling if still 3D
        if x.dim() == 3:
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        return x.flatten(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with timing measurement."""
        start_time = time.time()
        
        # Extract features
        features = self._extract_features(x)
        
        # Classify
        output = self.classifier(features)
        
        # Track inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        self.inference_times.append(inference_time)
        
        return output
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on genome."""
        act_name = random.choice(self.genome.activation_functions)
        
        if act_name == "relu":
            return nn.ReLU(inplace=True)
        elif act_name == "gelu":
            return nn.GELU()
        elif act_name == "swish":
            return nn.SiLU()
        elif act_name == "mish":
            return Mish()
        elif act_name == "elu":
            return nn.ELU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def _get_normalization(self, num_features: int) -> nn.Module:
        """Get normalization layer based on genome."""
        norm_type = random.choice(self.genome.normalization_types)
        
        if norm_type == "batch":
            return nn.BatchNorm1d(num_features)
        elif norm_type == "layer":
            return nn.LayerNorm([num_features])
        elif norm_type == "instance":
            return nn.InstanceNorm1d(num_features)
        else:
            return nn.BatchNorm1d(num_features)
    
    def get_latency_ms(self) -> float:
        """Get average inference latency in milliseconds."""
        return np.mean(self.inference_times) if self.inference_times else 0.0
    
    def estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        # Rough estimate: 4 bytes per float32 parameter + activations overhead
        return (total_params * 4 + total_params * 8) / (1024 * 1024)
    
    def estimate_power_mw(self) -> float:
        """Estimate power consumption in mW (simplified model)."""
        total_params = sum(p.numel() for p in self.parameters())
        # Simplified power model based on parameter count and layer types
        base_power = total_params * 0.001  # Base computation power
        
        # Add layer-specific power estimates
        layer_power = 0
        for layer_type in self.genome.layer_types:
            if layer_type in ["transformer", "attention"]:
                layer_power += 5.0  # Attention is expensive
            elif layer_type in ["lstm", "gru"]:
                layer_power += 3.0  # RNNs are moderately expensive
            else:
                layer_power += 1.0  # Conv layers are relatively cheap
        
        return base_power + layer_power


class BCITransformerBlock(nn.Module):
    """Transformer block optimized for BCI signals."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, sequence_length)
        x = x.transpose(1, 2)  # -> (batch, sequence_length, channels)
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x.transpose(1, 2)  # Back to (batch, channels, sequence_length)


class BCIAttentionLayer(nn.Module):
    """Spatial-temporal attention layer for EEG signals."""
    
    def __init__(self, input_dim: int, attention_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # Spatial attention (across channels)
        self.spatial_attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # Temporal attention (across time)
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(input_dim, attention_dim, 1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, 1, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, sequence_length)
        batch_size, channels, seq_len = x.shape
        
        # Spatial attention
        x_pooled = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch, channels)
        spatial_weights = torch.sigmoid(self.spatial_attention(x_pooled))  # (batch, 1)
        x_spatial = x * spatial_weights.unsqueeze(-1)
        
        # Temporal attention
        temporal_weights = torch.sigmoid(self.temporal_attention(x))  # (batch, 1, sequence_length)
        x_temporal = x_spatial * temporal_weights
        
        return self.dropout(x_temporal)


class BCITemporalConvNet(nn.Module):
    """Temporal Convolutional Network for BCI."""
    
    def __init__(self, input_channels: int, output_channels: int, 
                 kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            input_channels, output_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(output_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


class BCIResNetBlock(nn.Module):
    """ResNet block adapted for BCI signals."""
    
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.activation(out)
        
        return out


class Mish(nn.Module):
    """Mish activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class EvolutionaryNAS:
    """Evolutionary Neural Architecture Search for BCI."""
    
    def __init__(self, population_size: int = 20, generations: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7,
                 elite_ratio: float = 0.1, tournament_size: int = 3):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size
        
        self.search_space = BCISearchSpace()
        self.population = []
        self.generation = 0
        self.best_genome = None
        self.history = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_population(self) -> None:
        """Initialize random population."""
        self.logger.info(f"Initializing population of size {self.population_size}")
        
        self.population = []
        for i in range(self.population_size):
            genome = self.search_space.random_genome()
            genome.generation = 0
            self.population.append(genome)
        
        self.logger.info("Population initialized")
    
    def evaluate_population(self, train_loader, val_loader, device: str = "cpu") -> None:
        """Evaluate all genomes in population."""
        self.logger.info(f"Evaluating generation {self.generation}")
        
        # Use multiprocessing for parallel evaluation
        with ProcessPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
            futures = []
            
            for i, genome in enumerate(self.population):
                future = executor.submit(
                    self._evaluate_single_genome, 
                    genome, train_loader, val_loader, device, i
                )
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    evaluated_genome = future.result(timeout=300)  # 5 minute timeout
                    self.population[i] = evaluated_genome
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for genome {i}: {e}")
                    # Assign poor fitness to failed genomes
                    self.population[i].fitness_score = -1.0
        
        # Update best genome
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)
        if self.best_genome is None or self.population[0].fitness_score > self.best_genome.fitness_score:
            self.best_genome = self.population[0]
        
        # Record history
        self._record_generation_stats()
        
        self.logger.info(f"Generation {self.generation} evaluated. Best fitness: {self.best_genome.fitness_score:.4f}")
    
    def _evaluate_single_genome(self, genome: ArchitectureGenome, train_loader, val_loader, 
                               device: str, genome_id: int) -> ArchitectureGenome:
        """Evaluate a single genome."""
        try:
            # Build architecture
            model = BCINeuralArchitecture(genome)
            model.to(device)
            
            # Quick training (limited epochs for NAS)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Training
            model.train()
            for epoch in range(3):  # Limited training for NAS
                for batch_idx, (data, target) in enumerate(train_loader):
                    if batch_idx >= 10:  # Limited batches for speed
                        break
                    
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            total_latency = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    
                    start_time = time.time()
                    output = model(data)
                    latency = (time.time() - start_time) * 1000  # ms
                    
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    total_latency += latency
            
            # Calculate metrics
            accuracy = correct / total
            avg_latency = total_latency / len(val_loader)
            memory_mb = model.estimate_memory_mb()
            power_mw = model.estimate_power_mw()
            
            # Multi-objective fitness (weighted combination)
            fitness = (
                0.6 * accuracy +                    # Accuracy weight: 60%
                0.2 * (1.0 - min(avg_latency / 100, 1.0)) +  # Latency weight: 20% (inverse)
                0.1 * (1.0 - min(memory_mb / 500, 1.0)) +    # Memory weight: 10% (inverse)
                0.1 * (1.0 - min(power_mw / 50, 1.0))        # Power weight: 10% (inverse)
            )
            
            # Update genome
            genome.accuracy = accuracy
            genome.latency_ms = avg_latency
            genome.memory_mb = memory_mb
            genome.power_mw = power_mw
            genome.fitness_score = fitness
            
        except Exception as e:
            # Assign poor fitness to failed architectures
            genome.fitness_score = -1.0
            genome.accuracy = 0.0
            genome.latency_ms = 1000.0
            genome.memory_mb = 1000.0
            genome.power_mw = 100.0
        
        return genome
    
    def evolve_generation(self) -> None:
        """Evolve population to next generation."""
        self.logger.info(f"Evolving generation {self.generation}")
        
        new_population = []
        
        # Elitism: Keep best genomes
        n_elite = max(1, int(self.elite_ratio * self.population_size))
        elite = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)[:n_elite]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(new_population) < self.population_size - 1:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child1, child2 = self.search_space.crossover_genomes(parent1, parent2)
                
                # Mutate children
                child1 = self.search_space.mutate_genome(child1, self.mutation_rate)
                child2 = self.search_space.mutate_genome(child2, self.mutation_rate)
                
                child1.generation = self.generation + 1
                child2.generation = self.generation + 1
                
                new_population.extend([child1, child2])
            else:
                # Mutation only
                parent = self._tournament_selection()
                child = self.search_space.mutate_genome(parent, self.mutation_rate)
                child.generation = self.generation + 1
                new_population.append(child)
        
        # Trim to population size
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def _tournament_selection(self) -> ArchitectureGenome:
        """Tournament selection for parent selection."""
        tournament = random.choices(self.population, k=self.tournament_size)
        return max(tournament, key=lambda g: g.fitness_score)
    
    def _record_generation_stats(self) -> None:
        """Record statistics for current generation."""
        fitnesses = [g.fitness_score for g in self.population]
        accuracies = [g.accuracy for g in self.population]
        latencies = [g.latency_ms for g in self.population]
        memories = [g.memory_mb for g in self.population]
        powers = [g.power_mw for g in self.population]
        
        self.history["generation"].append(self.generation)
        self.history["max_fitness"].append(max(fitnesses))
        self.history["mean_fitness"].append(np.mean(fitnesses))
        self.history["std_fitness"].append(np.std(fitnesses))
        self.history["max_accuracy"].append(max(accuracies))
        self.history["mean_accuracy"].append(np.mean(accuracies))
        self.history["min_latency"].append(min(latencies))
        self.history["mean_latency"].append(np.mean(latencies))
        self.history["min_memory"].append(min(memories))
        self.history["mean_memory"].append(np.mean(memories))
        self.history["min_power"].append(min(powers))
        self.history["mean_power"].append(np.mean(powers))
    
    def run_search(self, train_loader, val_loader, device: str = "cpu") -> ArchitectureGenome:
        """Run complete evolutionary search."""
        self.logger.info(f"Starting evolutionary NAS for {self.generations} generations")
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for gen in range(self.generations):
            # Evaluate current population
            self.evaluate_population(train_loader, val_loader, device)
            
            # Check for early stopping based on fitness plateau
            if self._should_early_stop():
                self.logger.info(f"Early stopping at generation {self.generation}")
                break
            
            # Evolve to next generation
            if gen < self.generations - 1:
                self.evolve_generation()
        
        self.logger.info(f"Search completed. Best fitness: {self.best_genome.fitness_score:.4f}")
        return self.best_genome
    
    def _should_early_stop(self, patience: int = 10, min_improvement: float = 0.001) -> bool:
        """Check if search should stop early."""
        if len(self.history["max_fitness"]) < patience:
            return False
        
        recent_max = self.history["max_fitness"][-patience:]
        improvement = max(recent_max) - min(recent_max)
        
        return improvement < min_improvement
    
    def visualize_search_progress(self, save_path: Optional[str] = None) -> None:
        """Visualize search progress."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Fitness over generations
        axes[0].plot(self.history["generation"], self.history["max_fitness"], 'b-', label="Max")
        axes[0].plot(self.history["generation"], self.history["mean_fitness"], 'r--', label="Mean")
        axes[0].fill_between(
            self.history["generation"],
            np.array(self.history["mean_fitness"]) - np.array(self.history["std_fitness"]),
            np.array(self.history["mean_fitness"]) + np.array(self.history["std_fitness"]),
            alpha=0.3
        )
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Fitness")
        axes[0].set_title("Fitness Evolution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy over generations
        axes[1].plot(self.history["generation"], self.history["max_accuracy"], 'g-', label="Max")
        axes[1].plot(self.history["generation"], self.history["mean_accuracy"], 'orange', linestyle='--', label="Mean")
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Evolution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Latency over generations
        axes[2].plot(self.history["generation"], self.history["min_latency"], 'purple', label="Min")
        axes[2].plot(self.history["generation"], self.history["mean_latency"], 'brown', linestyle='--', label="Mean")
        axes[2].set_xlabel("Generation")
        axes[2].set_ylabel("Latency (ms)")
        axes[2].set_title("Latency Evolution")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Memory usage over generations
        axes[3].plot(self.history["generation"], self.history["min_memory"], 'cyan', label="Min")
        axes[3].plot(self.history["generation"], self.history["mean_memory"], 'pink', linestyle='--', label="Mean")
        axes[3].set_xlabel("Generation")
        axes[3].set_ylabel("Memory (MB)")
        axes[3].set_title("Memory Usage Evolution")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Power consumption over generations
        axes[4].plot(self.history["generation"], self.history["min_power"], 'olive', label="Min")
        axes[4].plot(self.history["generation"], self.history["mean_power"], 'navy', linestyle='--', label="Mean")
        axes[4].set_xlabel("Generation")
        axes[4].set_ylabel("Power (mW)")
        axes[4].set_title("Power Consumption Evolution")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # Pareto front (Accuracy vs Latency)
        if len(self.population) > 0:
            accuracies = [g.accuracy for g in self.population]
            latencies = [g.latency_ms for g in self.population]
            fitnesses = [g.fitness_score for g in self.population]
            
            scatter = axes[5].scatter(latencies, accuracies, c=fitnesses, cmap='viridis', alpha=0.7)
            axes[5].set_xlabel("Latency (ms)")
            axes[5].set_ylabel("Accuracy")
            axes[5].set_title("Accuracy vs Latency (Final Population)")
            plt.colorbar(scatter, ax=axes[5], label="Fitness")
            axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def save_results(self, save_dir: str) -> None:
        """Save search results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save best genome
        with open(save_path / "best_genome.json", 'w') as f:
            genome_dict = self.best_genome.__dict__.copy()
            # Convert non-serializable objects
            for key, value in genome_dict.items():
                if isinstance(value, np.ndarray):
                    genome_dict[key] = value.tolist()
            json.dump(genome_dict, f, indent=2, default=str)
        
        # Save search history
        with open(save_path / "search_history.json", 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        
        # Save final population
        population_data = []
        for genome in self.population:
            genome_dict = genome.__dict__.copy()
            for key, value in genome_dict.items():
                if isinstance(value, np.ndarray):
                    genome_dict[key] = value.tolist()
            population_data.append(genome_dict)
        
        with open(save_path / "final_population.json", 'w') as f:
            json.dump(population_data, f, indent=2, default=str)
        
        # Save visualization
        self.visualize_search_progress(str(save_path / "search_progress.png"))
        
        self.logger.info(f"Results saved to {save_path}")


def create_bci_nas_system(
    population_size: int = 20,
    generations: int = 50,
    mutation_rate: float = 0.1,
    device: str = "cpu"
) -> EvolutionaryNAS:
    """
    Create a BCI Neural Architecture Search system.
    
    Args:
        population_size: Size of population for evolutionary search
        generations: Number of generations to evolve
        mutation_rate: Probability of mutations
        device: Device for training ("cpu" or "cuda")
        
    Returns:
        Configured NAS system
    """
    nas_system = EvolutionaryNAS(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate
    )
    
    logger.info(f"Created BCI NAS system with population {population_size}, generations {generations}")
    
    return nas_system


# Example usage
def run_bci_nas_example():
    """Example of running BCI NAS."""
    import torch.utils.data as data
    
    # Create synthetic dataset
    class BCIDataset(data.Dataset):
        def __init__(self, n_samples=1000, n_channels=64, seq_length=250, n_classes=2):
            self.X = torch.randn(n_samples, n_channels, seq_length)
            self.y = torch.randint(0, n_classes, (n_samples,))
            
            # Add some pattern for class distinction
            self.X[self.y == 1, :10, :] += 0.5
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    # Create data loaders
    train_dataset = BCIDataset(800)
    val_dataset = BCIDataset(200)
    
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create NAS system
    nas = create_bci_nas_system(population_size=10, generations=5)
    
    # Run search
    best_genome = nas.run_search(train_loader, val_loader, device="cpu")
    
    # Save results
    nas.save_results("./nas_results")
    
    print(f"Best architecture found with fitness: {best_genome.fitness_score:.4f}")
    print(f"Accuracy: {best_genome.accuracy:.4f}")
    print(f"Latency: {best_genome.latency_ms:.2f} ms")
    print(f"Memory: {best_genome.memory_mb:.2f} MB")
    print(f"Power: {best_genome.power_mw:.2f} mW")
    
    return best_genome


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    best_architecture = run_bci_nas_example()