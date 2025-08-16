"""
Multi-Modal Fusion Transformers for Advanced BCI Processing.

This module implements breakthrough multi-modal fusion transformers that combine:
- Neural signals (EEG, ECoG, fMRI, MEG)
- Behavioral data (eye tracking, facial expressions, gestures)
- Contextual information (task context, environment, user state)
- Temporal dynamics (short-term, long-term patterns)

Research Contributions:
1. Cross-Modal Attention Mechanisms for neural-behavioral fusion
2. Hierarchical Temporal Fusion for multi-scale time dynamics
3. Adaptive Modal Weighting based on signal quality and relevance
4. Causal-Aware Fusion to understand cause-effect relationships
5. Uncertainty-Aware Predictions with confidence estimation

Novel Architectures:
- MultiModalTransformer: Core fusion architecture
- CrossModalAttention: Attention across different modalities  
- TemporalHierarchyFusion: Multi-scale temporal processing
- AdaptiveModalWeighting: Dynamic modality importance
- CausalFusionNetwork: Causal relationship modeling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
import time
import math
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModalityConfig:
    """Configuration for individual modalities."""
    
    name: str
    input_dim: int
    sequence_length: int
    sampling_rate: float
    processing_type: str = "temporal"  # "temporal", "spatial", "spectral", "hybrid"
    
    # Quality metrics
    signal_quality_threshold: float = 0.7
    noise_level: float = 0.1
    reliability_score: float = 1.0
    
    # Processing parameters
    normalize: bool = True
    apply_filtering: bool = True
    extract_features: bool = True
    
    # Attention parameters
    attention_dim: int = 64
    n_attention_heads: int = 8
    
    # Temporal parameters
    temporal_windows: List[int] = field(default_factory=lambda: [50, 100, 250, 500])
    temporal_strides: List[int] = field(default_factory=lambda: [10, 25, 50, 100])


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""
    
    # Fusion strategy
    fusion_type: str = "hierarchical"  # "early", "late", "hierarchical", "adaptive"
    fusion_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    
    # Attention mechanisms
    use_cross_modal_attention: bool = True
    use_self_attention: bool = True
    use_temporal_attention: bool = True
    
    # Hierarchical fusion
    hierarchy_levels: int = 3
    level_dimensions: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # Adaptive weighting
    use_adaptive_weighting: bool = True
    weighting_network_dim: int = 64
    
    # Causal modeling
    use_causal_attention: bool = True
    causal_window_size: int = 100
    
    # Uncertainty estimation
    use_uncertainty: bool = True
    uncertainty_type: str = "aleatoric"  # "aleatoric", "epistemic", "both"
    
    # Output configuration
    output_dim: int = 64
    n_classes: int = 2


class MultiModalEmbedding(nn.Module):
    """Multi-modal embedding layer with modality-specific processing."""
    
    def __init__(self, modality_configs: List[ModalityConfig], fusion_config: FusionConfig):
        super().__init__()
        
        self.modality_configs = {config.name: config for config in modality_configs}
        self.fusion_config = fusion_config
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleDict()
        self.modality_projections = nn.ModuleDict()
        
        for config in modality_configs:
            # Build modality-specific encoder
            encoder = self._build_modality_encoder(config)
            self.modality_encoders[config.name] = encoder
            
            # Project to common dimension
            projection = nn.Linear(config.attention_dim, fusion_config.output_dim)
            self.modality_projections[config.name] = projection
        
        # Positional encoding for temporal alignment
        self.positional_encoding = PositionalEncoding(fusion_config.output_dim, max_len=5000)
        
        # Modality type embeddings
        self.modality_type_embeddings = nn.Embedding(
            len(modality_configs), fusion_config.output_dim
        )
        
    def _build_modality_encoder(self, config: ModalityConfig) -> nn.Module:
        """Build encoder specific to modality type."""
        
        if config.processing_type == "temporal":
            return TemporalEncoder(
                input_dim=config.input_dim,
                hidden_dim=config.attention_dim,
                sequence_length=config.sequence_length,
                n_heads=config.n_attention_heads
            )
        
        elif config.processing_type == "spatial":
            return SpatialEncoder(
                input_dim=config.input_dim,
                hidden_dim=config.attention_dim,
                n_heads=config.n_attention_heads
            )
        
        elif config.processing_type == "spectral":
            return SpectralEncoder(
                input_dim=config.input_dim,
                hidden_dim=config.attention_dim,
                sampling_rate=config.sampling_rate
            )
        
        elif config.processing_type == "hybrid":
            return HybridEncoder(
                input_dim=config.input_dim,
                hidden_dim=config.attention_dim,
                sequence_length=config.sequence_length,
                sampling_rate=config.sampling_rate,
                n_heads=config.n_attention_heads
            )
        
        else:
            # Default temporal encoder
            return TemporalEncoder(
                input_dim=config.input_dim,
                hidden_dim=config.attention_dim,
                sequence_length=config.sequence_length,
                n_heads=config.n_attention_heads
            )
    
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-modal embedding.
        
        Args:
            modal_inputs: Dictionary mapping modality names to input tensors
            
        Returns:
            Dictionary of embedded modality representations
        """
        embedded_modalities = {}
        
        for modality_name, input_tensor in modal_inputs.items():
            if modality_name not in self.modality_encoders:
                logger.warning(f"Unknown modality: {modality_name}")
                continue
            
            # Encode modality-specific features
            encoded = self.modality_encoders[modality_name](input_tensor)
            
            # Project to common dimension
            projected = self.modality_projections[modality_name](encoded)
            
            # Add positional encoding
            if projected.dim() == 3:  # (batch, seq, features)
                projected = self.positional_encoding(projected)
            
            # Add modality type embedding
            modality_idx = list(self.modality_configs.keys()).index(modality_name)
            modality_type_emb = self.modality_type_embeddings(
                torch.tensor(modality_idx, device=projected.device)
            )
            
            if projected.dim() == 3:
                modality_type_emb = modality_type_emb.unsqueeze(0).unsqueeze(0).expand(
                    projected.size(0), projected.size(1), -1
                )
            else:
                modality_type_emb = modality_type_emb.unsqueeze(0).expand(
                    projected.size(0), -1
                )
            
            projected = projected + modality_type_emb
            
            embedded_modalities[modality_name] = projected
        
        return embedded_modalities


class TemporalEncoder(nn.Module):
    """Temporal encoder for time-series modalities."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sequence_length: int, n_heads: int = 8):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence, input_dim)
        x = self.input_projection(x)
        
        # Multi-scale temporal convolutions
        x_conv = x.transpose(1, 2)  # (batch, hidden_dim, sequence)
        conv_outputs = []
        for conv in self.temporal_convs:
            conv_out = F.gelu(conv(x_conv))
            conv_outputs.append(conv_out)
        
        # Concatenate and project back
        x_multi_scale = torch.cat(conv_outputs, dim=1)  # Concatenate along channel dim
        x_multi_scale = x_multi_scale.transpose(1, 2)  # Back to (batch, sequence, channels)
        
        # Residual connection
        x = x + x_multi_scale
        x = self.norm1(x)
        
        # Temporal attention
        attn_out, _ = self.temporal_attention(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = x + ffn_out
        
        return x


class SpatialEncoder(nn.Module):
    """Spatial encoder for spatial modalities (e.g., EEG channel locations)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 8):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Spatial convolutions (treating channels as spatial dimensions)
        self.spatial_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True
        )
        
        # Graph convolution for electrode connectivity
        self.graph_conv = GraphConvolution(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, features) or (batch, sequence, channels, features)
        original_shape = x.shape
        
        if x.dim() == 4:  # (batch, sequence, channels, features)
            batch, seq, channels, features = x.shape
            x = x.view(batch * seq, channels, features)
        
        x = self.input_projection(x)
        
        # Spatial convolution
        x_conv = x.transpose(1, 2)  # (batch, features, channels)
        x_conv = F.gelu(self.spatial_conv(x_conv))
        x_conv = x_conv.transpose(1, 2)  # Back to (batch, channels, features)
        
        # Spatial attention
        attn_out, _ = self.spatial_attention(x, x, x)
        x = x + attn_out
        
        # Graph convolution (simplified - assumes all-to-all connectivity)
        x = self.graph_conv(x)
        x = self.norm(x)
        
        # Reshape back if needed
        if len(original_shape) == 4:
            x = x.view(batch, seq, channels, -1)
        
        return x


class SpectralEncoder(nn.Module):
    """Spectral encoder for frequency domain features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sampling_rate: float):
        super().__init__()
        
        self.sampling_rate = sampling_rate
        self.hidden_dim = hidden_dim
        
        # Frequency band definitions
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Spectral feature extractors
        self.spectral_projections = nn.ModuleDict({
            band: nn.Linear(input_dim, hidden_dim // len(self.freq_bands))
            for band in self.freq_bands.keys()
        })
        
        # Spectral attention
        self.spectral_attention = nn.MultiheadAttention(
            hidden_dim, 8, batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # Extract spectral features for each frequency band
        spectral_features = []
        
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Apply bandpass filter (simplified)
            # In practice, you'd use proper signal processing
            band_features = self._extract_band_features(x, low_freq, high_freq)
            
            # Project to hidden dimension
            projected = self.spectral_projections[band_name](band_features)
            spectral_features.append(projected)
        
        # Concatenate frequency bands
        x_spectral = torch.cat(spectral_features, dim=-1)
        
        # Spectral attention
        attn_out, _ = self.spectral_attention(x_spectral, x_spectral, x_spectral)
        x_spectral = x_spectral + attn_out
        x_spectral = self.norm(x_spectral)
        
        return x_spectral
    
    def _extract_band_features(self, x: torch.Tensor, low_freq: float, high_freq: float) -> torch.Tensor:
        """Extract features for specific frequency band."""
        # Simplified spectral feature extraction
        # In practice, you'd use proper FFT and filtering
        
        # Apply simple moving average as frequency filter approximation
        kernel_size = min(10, x.size(1) // 4)
        if kernel_size > 1:
            x_filtered = F.avg_pool1d(
                x.transpose(1, 2), 
                kernel_size=kernel_size, 
                stride=1, 
                padding=kernel_size//2
            ).transpose(1, 2)
        else:
            x_filtered = x
        
        return x_filtered


class HybridEncoder(nn.Module):
    """Hybrid encoder combining temporal, spatial, and spectral processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sequence_length: int, 
                 sampling_rate: float, n_heads: int = 8):
        super().__init__()
        
        # Individual encoders
        self.temporal_encoder = TemporalEncoder(input_dim, hidden_dim//3, sequence_length, n_heads//2)
        self.spatial_encoder = SpatialEncoder(input_dim, hidden_dim//3, n_heads//2)
        self.spectral_encoder = SpectralEncoder(input_dim, hidden_dim//3, sampling_rate)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process with each encoder
        temporal_features = self.temporal_encoder(x)
        
        # For spatial encoder, we need to handle the dimension appropriately
        if x.dim() == 3:  # (batch, sequence, features)
            # Treat sequence as spatial dimension
            spatial_features = self.spatial_encoder(x.transpose(1, 2)).transpose(1, 2)
        else:
            spatial_features = self.spatial_encoder(x)
        
        spectral_features = self.spectral_encoder(x)
        
        # Concatenate features
        hybrid_features = torch.cat([temporal_features, spatial_features, spectral_features], dim=-1)
        
        # Fusion
        fused_features = self.fusion_layer(hybrid_features)
        fused_features = self.norm(fused_features)
        
        return fused_features


class GraphConvolution(nn.Module):
    """Graph convolution layer for spatial relationships."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Simplified graph convolution
        # In practice, you'd use the actual adjacency matrix of electrode positions
        
        if adjacency is None:
            # Use simple all-to-all connectivity
            adjacency = torch.ones(x.size(1), x.size(1), device=x.device)
            adjacency = adjacency / adjacency.sum(dim=1, keepdim=True)  # Normalize
        
        # Graph convolution: A * X * W
        x_transformed = self.linear(x)  # Apply weights
        x_conv = torch.bmm(adjacency.unsqueeze(0).expand(x.size(0), -1, -1), x_transformed)
        
        return self.activation(x_conv)


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for modality fusion."""
    
    def __init__(self, embed_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        # Query, Key, Value projections for each modality
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Modal compatibility matrix
        self.modal_compatibility = nn.Parameter(torch.randn(1, n_heads, 1, 1))
        
    def forward(self, query_modality: torch.Tensor, key_modality: torch.Tensor, 
                value_modality: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Cross-modal attention computation.
        
        Args:
            query_modality: Query modality tensor (batch, seq_len, embed_dim)
            key_modality: Key modality tensor (batch, seq_len, embed_dim)  
            value_modality: Value modality tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Attended features (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = query_modality.shape
        head_dim = embed_dim // self.n_heads
        
        # Linear projections
        Q = self.q_linear(query_modality).view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        K = self.k_linear(key_modality).view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        V = self.v_linear(value_modality).view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with modal compatibility
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores * self.modal_compatibility  # Apply compatibility weights
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection and residual connection
        output = self.out_linear(attended)
        output = self.norm(output + query_modality)
        
        return output


class TemporalHierarchyFusion(nn.Module):
    """Hierarchical temporal fusion across multiple time scales."""
    
    def __init__(self, embed_dim: int, hierarchy_levels: int = 3, 
                 level_dimensions: List[int] = None):
        super().__init__()
        
        self.hierarchy_levels = hierarchy_levels
        
        if level_dimensions is None:
            level_dimensions = [embed_dim // (2**i) for i in range(hierarchy_levels)]
        
        self.level_dimensions = level_dimensions
        
        # Hierarchical processing layers
        self.level_processors = nn.ModuleList()
        self.level_projections = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        
        for i, dim in enumerate(level_dimensions):
            # Temporal processing at this level
            processor = nn.Sequential(
                nn.Conv1d(embed_dim if i == 0 else level_dimensions[i-1], dim, 
                         kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(dim),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(dim)
            )
            self.level_processors.append(processor)
            
            # Projection layer
            projection = nn.Linear(dim, embed_dim)
            self.level_projections.append(projection)
            
            # Upsampling for lower levels
            if i > 0:
                upsampling = nn.ConvTranspose1d(dim, embed_dim, kernel_size=2**i, stride=2**i)
                self.upsampling_layers.append(upsampling)
        
        # Final fusion layer
        self.fusion_layer = nn.Linear(embed_dim * hierarchy_levels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical temporal fusion.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Fused temporal features (batch, seq_len, embed_dim)
        """
        # x shape: (batch, seq_len, embed_dim)
        x_conv = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        level_outputs = []
        current_input = x_conv
        
        for i, (processor, projection) in enumerate(zip(self.level_processors, self.level_projections)):
            # Process at current level
            level_out = processor(current_input)
            
            # Downsample for next level
            if i < self.hierarchy_levels - 1:
                current_input = F.avg_pool1d(level_out, kernel_size=2, stride=2)
            
            # Project back to common dimension and upsample if needed
            level_out_proj = level_out.transpose(1, 2)  # (batch, seq_len/scale, dim)
            level_out_proj = projection(level_out_proj)  # (batch, seq_len/scale, embed_dim)
            
            # Upsample to original sequence length
            if i > 0:
                level_out_proj = level_out_proj.transpose(1, 2)  # (batch, embed_dim, seq_len/scale)
                level_out_proj = self.upsampling_layers[i-1](level_out_proj)
                level_out_proj = level_out_proj.transpose(1, 2)  # (batch, seq_len, embed_dim)
                
                # Trim or pad to match original sequence length
                target_len = x.size(1)
                current_len = level_out_proj.size(1)
                if current_len > target_len:
                    level_out_proj = level_out_proj[:, :target_len, :]
                elif current_len < target_len:
                    padding = target_len - current_len
                    level_out_proj = F.pad(level_out_proj, (0, 0, 0, padding))
            
            level_outputs.append(level_out_proj)
        
        # Concatenate all levels
        fused_features = torch.cat(level_outputs, dim=-1)
        
        # Final fusion
        output = self.fusion_layer(fused_features)
        output = self.norm(output)
        
        return output


class AdaptiveModalWeighting(nn.Module):
    """Adaptive weighting network for dynamic modality importance."""
    
    def __init__(self, modality_names: List[str], embed_dim: int, 
                 weighting_network_dim: int = 64):
        super().__init__()
        
        self.modality_names = modality_names
        self.n_modalities = len(modality_names)
        self.embed_dim = embed_dim
        
        # Context network to compute modality weights
        self.context_network = nn.Sequential(
            nn.Linear(embed_dim * self.n_modalities, weighting_network_dim),
            nn.GELU(),
            nn.Linear(weighting_network_dim, weighting_network_dim),
            nn.GELU(),
            nn.Linear(weighting_network_dim, self.n_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Quality assessment networks for each modality
        self.quality_networks = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed_dim, weighting_network_dim // 2),
                nn.GELU(),
                nn.Linear(weighting_network_dim // 2, 1),
                nn.Sigmoid()
            ) for name in modality_names
        })
        
        # Temporal consistency network
        self.temporal_consistency = nn.LSTM(
            self.n_modalities, weighting_network_dim, batch_first=True
        )
        self.consistency_projection = nn.Linear(weighting_network_dim, self.n_modalities)
        
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive weights for modalities.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Weighted modality features
        """
        batch_size = list(modality_features.values())[0].size(0)
        seq_len = list(modality_features.values())[0].size(1)
        
        # Ensure all modalities are present
        available_modalities = []
        features_list = []
        
        for name in self.modality_names:
            if name in modality_features:
                available_modalities.append(name)
                # Global average pooling for context
                feat = modality_features[name]  # (batch, seq_len, embed_dim)
                global_feat = feat.mean(dim=1)  # (batch, embed_dim)
                features_list.append(global_feat)
            else:
                # Use zero features for missing modalities
                zero_feat = torch.zeros(batch_size, self.embed_dim, 
                                      device=list(modality_features.values())[0].device)
                features_list.append(zero_feat)
        
        # Concatenate global features
        global_context = torch.cat(features_list, dim=-1)  # (batch, embed_dim * n_modalities)
        
        # Compute context-based weights
        context_weights = self.context_network(global_context)  # (batch, n_modalities)
        
        # Compute quality-based weights
        quality_weights = []
        for i, name in enumerate(self.modality_names):
            if name in modality_features:
                quality = self.quality_networks[name](features_list[i])  # (batch, 1)
                quality_weights.append(quality)
            else:
                # Low quality for missing modalities
                quality_weights.append(torch.zeros(batch_size, 1, 
                                                 device=context_weights.device))
        
        quality_weights = torch.cat(quality_weights, dim=-1)  # (batch, n_modalities)
        
        # Temporal consistency
        weight_history = context_weights.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, n_modalities)
        consistency_out, _ = self.temporal_consistency(weight_history)
        consistency_weights = torch.sigmoid(self.consistency_projection(consistency_out))
        
        # Combine all weight sources
        final_weights = context_weights.unsqueeze(1) * quality_weights.unsqueeze(1) * consistency_weights
        final_weights = F.softmax(final_weights, dim=-1)  # (batch, seq_len, n_modalities)
        
        # Apply weights to modality features
        weighted_features = {}
        for i, name in enumerate(self.modality_names):
            if name in modality_features:
                weight = final_weights[:, :, i:i+1]  # (batch, seq_len, 1)
                weighted_features[name] = modality_features[name] * weight
            
        return weighted_features


class CausalFusionNetwork(nn.Module):
    """Causal fusion network for understanding causal relationships."""
    
    def __init__(self, embed_dim: int, causal_window_size: int = 100, 
                 n_causal_layers: int = 3):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.causal_window_size = causal_window_size
        self.n_causal_layers = n_causal_layers
        
        # Causal convolution layers
        self.causal_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=2**i)
            for i in range(n_causal_layers)
        ])
        
        # Causal attention with temporal masking
        self.causal_attention = nn.MultiheadAttention(
            embed_dim, 8, batch_first=True
        )
        
        # Granger causality estimation network
        self.granger_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_causal_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Causal fusion with causality estimation.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Tuple of (causal_features, causality_weights)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, x.device)
        
        # Causal convolutions
        causal_features = x
        for i, (conv, norm) in enumerate(zip(self.causal_convs, self.norm_layers)):
            # Apply causal convolution
            conv_input = causal_features.transpose(1, 2)  # (batch, embed_dim, seq_len)
            conv_out = conv(conv_input)
            conv_out = conv_out.transpose(1, 2)  # (batch, seq_len, embed_dim)
            
            # Residual connection and normalization
            causal_features = norm(causal_features + conv_out)
        
        # Causal attention
        attn_out, attn_weights = self.causal_attention(
            causal_features, causal_features, causal_features, 
            attn_mask=causal_mask
        )
        causal_features = causal_features + attn_out
        
        # Estimate causality strengths
        causality_weights = self._estimate_causality(causal_features)
        
        return causal_features, causality_weights
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _estimate_causality(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate causality strengths using simplified Granger causality."""
        batch_size, seq_len, embed_dim = features.shape
        
        # Create lagged features
        causality_scores = []
        
        for lag in range(1, min(10, seq_len // 2)):  # Check various lags
            # Current features
            current_features = features[:, lag:, :]  # (batch, seq_len-lag, embed_dim)
            
            # Lagged features
            lagged_features = features[:, :-lag, :]  # (batch, seq_len-lag, embed_dim)
            
            # Concatenate current and lagged
            combined_features = torch.cat([current_features, lagged_features], dim=-1)
            
            # Estimate causality strength
            causality = self.granger_network(combined_features)  # (batch, seq_len-lag, 1)
            
            # Pad to original length
            padded_causality = F.pad(causality, (0, 0, lag, 0))  # Pad at beginning
            causality_scores.append(padded_causality)
        
        # Average across lags
        causality_weights = torch.stack(causality_scores, dim=-1).mean(dim=-1)
        
        return causality_weights


class UncertaintyEstimation(nn.Module):
    """Uncertainty estimation for multi-modal predictions."""
    
    def __init__(self, embed_dim: int, uncertainty_type: str = "both"):
        super().__init__()
        
        self.uncertainty_type = uncertainty_type
        
        # Aleatoric uncertainty (data noise)
        if uncertainty_type in ["aleatoric", "both"]:
            self.aleatoric_network = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Softplus()  # Ensure positive values
            )
        
        # Epistemic uncertainty (model uncertainty)
        if uncertainty_type in ["epistemic", "both"]:
            self.epistemic_dropout = nn.Dropout(0.5)
            self.n_mc_samples = 10
        
    def forward(self, features: torch.Tensor, training: bool = False) -> Dict[str, torch.Tensor]:
        """
        Estimate prediction uncertainty.
        
        Args:
            features: Input features (batch, seq_len, embed_dim)
            training: Whether in training mode
            
        Returns:
            Dictionary with uncertainty estimates
        """
        uncertainty_estimates = {}
        
        # Aleatoric uncertainty
        if self.uncertainty_type in ["aleatoric", "both"]:
            aleatoric_var = self.aleatoric_network(features)  # (batch, seq_len, 1)
            uncertainty_estimates["aleatoric"] = aleatoric_var
        
        # Epistemic uncertainty (Monte Carlo Dropout)
        if self.uncertainty_type in ["epistemic", "both"] and training:
            mc_predictions = []
            for _ in range(self.n_mc_samples):
                # Apply dropout multiple times
                mc_features = self.epistemic_dropout(features)
                mc_predictions.append(mc_features)
            
            # Calculate variance across MC samples
            mc_stack = torch.stack(mc_predictions, dim=0)  # (n_samples, batch, seq_len, embed_dim)
            epistemic_var = torch.var(mc_stack, dim=0).mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
            uncertainty_estimates["epistemic"] = epistemic_var
        
        return uncertainty_estimates


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal alignment."""
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class MultiModalFusionTransformer(nn.Module):
    """Main multi-modal fusion transformer architecture."""
    
    def __init__(self, modality_configs: List[ModalityConfig], 
                 fusion_config: FusionConfig):
        super().__init__()
        
        self.modality_configs = modality_configs
        self.fusion_config = fusion_config
        self.modality_names = [config.name for config in modality_configs]
        
        # Multi-modal embedding
        self.modal_embedding = MultiModalEmbedding(modality_configs, fusion_config)
        
        # Cross-modal attention layers
        if fusion_config.use_cross_modal_attention:
            self.cross_modal_attentions = nn.ModuleDict()
            for i, mod1 in enumerate(self.modality_names):
                for j, mod2 in enumerate(self.modality_names):
                    if i != j:  # Cross-modal only
                        key = f"{mod1}_to_{mod2}"
                        self.cross_modal_attentions[key] = CrossModalAttention(
                            fusion_config.output_dim, 8, dropout=0.1
                        )
        
        # Hierarchical temporal fusion
        if fusion_config.fusion_type == "hierarchical":
            self.temporal_hierarchy = TemporalHierarchyFusion(
                fusion_config.output_dim, 
                fusion_config.hierarchy_levels,
                fusion_config.level_dimensions
            )
        
        # Adaptive modal weighting
        if fusion_config.use_adaptive_weighting:
            self.adaptive_weighting = AdaptiveModalWeighting(
                self.modality_names, 
                fusion_config.output_dim,
                fusion_config.weighting_network_dim
            )
        
        # Causal fusion
        if fusion_config.use_causal_attention:
            self.causal_fusion = CausalFusionNetwork(
                fusion_config.output_dim,
                fusion_config.causal_window_size
            )
        
        # Uncertainty estimation
        if fusion_config.use_uncertainty:
            self.uncertainty_estimation = UncertaintyEstimation(
                fusion_config.output_dim,
                fusion_config.uncertainty_type
            )
        
        # Final fusion layers
        total_fusion_dim = fusion_config.output_dim * len(self.modality_names)
        self.fusion_layers = nn.ModuleList()
        
        current_dim = total_fusion_dim
        for layer_dim in fusion_config.fusion_layers:
            self.fusion_layers.append(nn.Linear(current_dim, layer_dim))
            self.fusion_layers.append(nn.GELU())
            self.fusion_layers.append(nn.Dropout(0.1))
            current_dim = layer_dim
        
        # Output layer
        self.output_layer = nn.Linear(current_dim, fusion_config.n_classes)
        
        # Performance tracking
        self.forward_times = []
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of multi-modal fusion transformer.
        
        Args:
            modal_inputs: Dictionary mapping modality names to input tensors
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        start_time = time.time()
        
        # Multi-modal embedding
        embedded_modalities = self.modal_embedding(modal_inputs)
        
        # Cross-modal attention
        if self.fusion_config.use_cross_modal_attention:
            attended_modalities = {}
            for mod_name, mod_features in embedded_modalities.items():
                # Aggregate attention from other modalities
                attended_features = mod_features
                attention_count = 0
                
                for other_mod in self.modality_names:
                    if other_mod != mod_name and other_mod in embedded_modalities:
                        attention_key = f"{other_mod}_to_{mod_name}"
                        if attention_key in self.cross_modal_attentions:
                            cross_attention = self.cross_modal_attentions[attention_key](
                                mod_features,  # Query
                                embedded_modalities[other_mod],  # Key
                                embedded_modalities[other_mod]   # Value
                            )
                            attended_features = attended_features + cross_attention
                            attention_count += 1
                
                if attention_count > 0:
                    attended_features = attended_features / (attention_count + 1)
                
                attended_modalities[mod_name] = attended_features
            
            embedded_modalities = attended_modalities
        
        # Adaptive modal weighting
        if self.fusion_config.use_adaptive_weighting:
            embedded_modalities = self.adaptive_weighting(embedded_modalities)
        
        # Hierarchical temporal fusion for each modality
        if self.fusion_config.fusion_type == "hierarchical":
            temporal_fused = {}
            for mod_name, mod_features in embedded_modalities.items():
                temporal_fused[mod_name] = self.temporal_hierarchy(mod_features)
            embedded_modalities = temporal_fused
        
        # Concatenate all modalities
        modality_features_list = []
        for mod_name in self.modality_names:
            if mod_name in embedded_modalities:
                modality_features_list.append(embedded_modalities[mod_name])
            else:
                # Use zero features for missing modalities
                zero_features = torch.zeros_like(list(embedded_modalities.values())[0])
                modality_features_list.append(zero_features)
        
        concatenated_features = torch.cat(modality_features_list, dim=-1)
        
        # Causal fusion
        causal_features = concatenated_features
        causality_weights = None
        if self.fusion_config.use_causal_attention:
            causal_features, causality_weights = self.causal_fusion(concatenated_features)
        
        # Global pooling for classification
        pooled_features = causal_features.mean(dim=1)  # (batch, embed_dim)
        
        # Final fusion layers
        fused_features = pooled_features
        for layer in self.fusion_layers:
            fused_features = layer(fused_features)
        
        # Output predictions
        logits = self.output_layer(fused_features)
        
        # Uncertainty estimation
        uncertainty_estimates = {}
        if self.fusion_config.use_uncertainty:
            uncertainty_estimates = self.uncertainty_estimation(
                causal_features, training=self.training
            )
        
        # Track inference time
        inference_time = time.time() - start_time
        self.forward_times.append(inference_time)
        
        # Prepare output
        output = {
            "logits": logits,
            "predictions": F.softmax(logits, dim=-1),
            "uncertainty": uncertainty_estimates,
            "causality_weights": causality_weights,
            "inference_time": inference_time
        }
        
        return output
    
    def get_average_inference_time(self) -> float:
        """Get average inference time in seconds."""
        return np.mean(self.forward_times) if self.forward_times else 0.0


def create_multimodal_bci_system(
    modality_configs: List[ModalityConfig],
    fusion_config: Optional[FusionConfig] = None
) -> MultiModalFusionTransformer:
    """
    Create a multi-modal BCI fusion system.
    
    Args:
        modality_configs: List of modality configurations
        fusion_config: Fusion configuration (optional)
        
    Returns:
        Configured multi-modal fusion transformer
    """
    if fusion_config is None:
        fusion_config = FusionConfig(
            fusion_type="hierarchical",
            use_cross_modal_attention=True,
            use_adaptive_weighting=True,
            use_causal_attention=True,
            use_uncertainty=True
        )
    
    model = MultiModalFusionTransformer(modality_configs, fusion_config)
    
    logger.info(f"Created multi-modal BCI system with {len(modality_configs)} modalities")
    
    return model


# Example usage
def create_example_multimodal_system():
    """Example of creating a multi-modal BCI system."""
    
    # Define modalities
    modality_configs = [
        ModalityConfig(
            name="eeg",
            input_dim=64,
            sequence_length=250,
            sampling_rate=250.0,
            processing_type="hybrid"
        ),
        ModalityConfig(
            name="eye_tracking",
            input_dim=4,  # x, y, pupil_diameter, blink
            sequence_length=250,
            sampling_rate=250.0,
            processing_type="temporal"
        ),
        ModalityConfig(
            name="behavioral",
            input_dim=10,  # Various behavioral metrics
            sequence_length=250,
            sampling_rate=250.0,
            processing_type="temporal"
        ),
        ModalityConfig(
            name="context",
            input_dim=20,  # Task context, environment variables
            sequence_length=1,  # Static context
            sampling_rate=1.0,
            processing_type="spatial"
        )
    ]
    
    # Create fusion configuration
    fusion_config = FusionConfig(
        fusion_type="hierarchical",
        fusion_layers=[512, 256, 128],
        use_cross_modal_attention=True,
        use_adaptive_weighting=True,
        use_causal_attention=True,
        use_uncertainty=True,
        output_dim=128,
        n_classes=4  # 4-class BCI task
    )
    
    # Create system
    multimodal_system = create_multimodal_bci_system(modality_configs, fusion_config)
    
    return multimodal_system


def run_multimodal_example():
    """Run example with synthetic data."""
    
    # Create system
    system = create_example_multimodal_system()
    
    # Create synthetic data
    batch_size = 8
    modal_inputs = {
        "eeg": torch.randn(batch_size, 250, 64),
        "eye_tracking": torch.randn(batch_size, 250, 4),
        "behavioral": torch.randn(batch_size, 250, 10),
        "context": torch.randn(batch_size, 1, 20)
    }
    
    # Forward pass
    with torch.no_grad():
        output = system(modal_inputs)
    
    print("Multi-Modal BCI System Output:")
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Inference time: {output['inference_time']:.4f} seconds")
    print(f"Predicted classes: {output['predictions'].argmax(dim=-1)}")
    
    if output['uncertainty']:
        for unc_type, unc_values in output['uncertainty'].items():
            print(f"{unc_type.capitalize()} uncertainty shape: {unc_values.shape}")
    
    return output


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    example_output = run_multimodal_example()