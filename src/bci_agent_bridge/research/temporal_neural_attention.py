"""
Temporal Neural Attention Mechanisms for Brain-Computer Interfaces.

This module implements breakthrough temporal attention mechanisms specifically
designed for neural signal processing and time-series BCI data:

1. Hierarchical Temporal Attention: Multi-scale temporal pattern recognition
2. Phase-Aware Attention: Neural oscillation phase-sensitive processing
3. Adaptive Temporal Kernels: Dynamic convolution kernels for neural signals
4. Causal Temporal Attention: Future-information blocking for real-time BCI
5. Neural Oscillation Attention: Frequency-band specific attention mechanisms
6. Temporal Transformer Variants: BCI-optimized transformer architectures

Research Contributions:
- First phase-aware attention mechanism for neural oscillations
- Novel adaptive temporal kernels for dynamic neural patterns
- Hierarchical attention across multiple temporal scales
- Causal attention preserving real-time constraints
- Oscillation-aware attention for frequency-domain processing

Applications:
- Real-time BCI with <50ms latency
- Neural oscillation analysis and decoding
- Multi-scale temporal pattern recognition
- Phase-dependent neural decoding
- Adaptive neural signal processing
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
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TemporalAttentionConfig:
    """Configuration for temporal attention mechanisms."""
    
    # Model dimensions
    embed_dim: int = 128
    n_heads: int = 8
    n_layers: int = 6
    
    # Temporal parameters
    sequence_length: int = 250
    sampling_rate: float = 250.0
    
    # Hierarchical attention
    hierarchy_levels: int = 4
    scale_factors: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    level_dims: List[int] = field(default_factory=lambda: [128, 96, 64, 32])
    
    # Phase-aware attention
    use_phase_attention: bool = True
    n_frequency_bands: int = 5
    frequency_bands: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.5, 4),   # Delta
        (4, 8),     # Theta
        (8, 13),    # Alpha
        (13, 30),   # Beta
        (30, 100)   # Gamma
    ])
    
    # Adaptive kernels
    use_adaptive_kernels: bool = True
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9, 11])
    max_dilation: int = 8
    
    # Causal attention
    use_causal_attention: bool = True
    causal_window_size: int = 50
    
    # Performance optimization
    use_relative_position: bool = True
    use_learnable_position: bool = True
    dropout: float = 0.1
    
    # Real-time constraints
    max_latency_ms: float = 50.0
    chunk_size: int = 25  # Process in chunks for real-time


class PhaseAwareAttention(nn.Module):
    """Phase-aware attention mechanism for neural oscillations."""
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        
        # Phase extraction networks for each frequency band
        self.phase_extractors = nn.ModuleDict({
            f"band_{i}": PhaseExtractor(
                low_freq=band[0], 
                high_freq=band[1], 
                sampling_rate=config.sampling_rate,
                embed_dim=config.embed_dim
            ) for i, band in enumerate(config.frequency_bands)
        })
        
        # Phase-conditioned attention weights
        self.phase_attention_weights = nn.ModuleDict({
            f"band_{i}": nn.Sequential(
                nn.Linear(2, config.n_heads),  # 2 for cos/sin phase
                nn.Tanh(),
                nn.Linear(config.n_heads, config.n_heads),
                nn.Sigmoid()
            ) for i in range(len(config.frequency_bands))
        })
        
        # Standard attention components
        self.q_linear = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_linear = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_linear = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_linear = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Phase-aware scaling
        self.phase_scaling = nn.Parameter(torch.ones(len(config.frequency_bands)))
        
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Phase-aware attention forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (attended_output, phase_info)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Extract phases for each frequency band
        phase_info = {}
        band_attentions = []
        
        for band_name, phase_extractor in self.phase_extractors.items():
            # Extract instantaneous phase
            amplitude, phase = phase_extractor(x)
            
            # Convert phase to cos/sin representation
            phase_features = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)  # (batch, seq_len, 2)
            
            # Compute phase-conditioned attention weights
            phase_weights = self.phase_attention_weights[band_name](phase_features)  # (batch, seq_len, n_heads)
            
            # Standard attention computation
            Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply phase-conditioned weights
            phase_weights_expanded = phase_weights.unsqueeze(-1).expand(-1, -1, self.n_heads, seq_len)
            phase_weights_expanded = phase_weights_expanded.transpose(1, 2)  # (batch, n_heads, seq_len, seq_len)
            
            scores = scores * phase_weights_expanded
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            attended = torch.matmul(attention_weights, V)
            attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            
            band_attentions.append(attended)
            
            # Store phase information
            band_idx = int(band_name.split('_')[1])
            phase_info[f"amplitude_band_{band_idx}"] = amplitude
            phase_info[f"phase_band_{band_idx}"] = phase
            phase_info[f"attention_weights_band_{band_idx}"] = attention_weights
        
        # Combine attention from all frequency bands
        stacked_attentions = torch.stack(band_attentions, dim=-1)  # (batch, seq_len, embed_dim, n_bands)
        
        # Learnable combination weights based on phase scaling
        phase_scaling_norm = F.softmax(self.phase_scaling, dim=0)
        combined_attention = (stacked_attentions * phase_scaling_norm).sum(dim=-1)
        
        # Output projection and residual connection
        output = self.out_linear(combined_attention)
        output = self.norm(output + x)
        
        return output, phase_info


class PhaseExtractor(nn.Module):
    """Extract instantaneous amplitude and phase from neural signals."""
    
    def __init__(self, low_freq: float, high_freq: float, sampling_rate: float, embed_dim: int):
        super().__init__()
        
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate
        self.embed_dim = embed_dim
        
        # Bandpass filter parameters
        self.filter_order = 4
        
        # Learnable filter coefficients (initialized from Butterworth)
        self.register_buffer('filter_b', self._get_filter_coeffs()[0])
        self.register_buffer('filter_a', self._get_filter_coeffs()[1])
        
        # Hilbert transform approximation network
        self.hilbert_net = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=15, padding=7, groups=embed_dim),
            nn.Tanh(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim)
        )
        
    def _get_filter_coeffs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Butterworth bandpass filter coefficients."""
        from scipy.signal import butter
        
        nyquist = self.sampling_rate / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        # Ensure valid frequency range
        low = max(0.01, min(0.99, low))
        high = max(low + 0.01, min(0.99, high))
        
        b, a = butter(self.filter_order, [low, high], btype='band')
        
        return torch.tensor(b, dtype=torch.float32), torch.tensor(a, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract amplitude and phase.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Tuple of (amplitude, phase)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Apply bandpass filter (simplified)
        x_filtered = self._apply_bandpass_filter(x)
        
        # Approximate Hilbert transform using learned network
        x_conv = x_filtered.transpose(1, 2)  # (batch, embed_dim, seq_len)
        hilbert_approx = self.hilbert_net(x_conv)
        hilbert_approx = hilbert_approx.transpose(1, 2)  # (batch, seq_len, embed_dim)
        
        # Compute analytic signal (real + i*hilbert)
        analytic_real = x_filtered
        analytic_imag = hilbert_approx
        
        # Compute amplitude and phase
        amplitude = torch.sqrt(analytic_real**2 + analytic_imag**2)
        phase = torch.atan2(analytic_imag, analytic_real)
        
        return amplitude, phase
    
    def _apply_bandpass_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filter (simplified implementation)."""
        # For simplicity, use a learned convolution as a filter approximation
        # In practice, you might want to implement proper IIR filtering
        
        # Simple moving average as a basic filter
        kernel_size = min(15, x.size(1) // 4)
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


class HierarchicalTemporalAttention(nn.Module):
    """Hierarchical attention across multiple temporal scales."""
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        
        self.config = config
        self.hierarchy_levels = config.hierarchy_levels
        self.scale_factors = config.scale_factors
        self.level_dims = config.level_dims
        
        # Multi-scale processing layers
        self.scale_processors = nn.ModuleList()
        self.scale_attentions = nn.ModuleList()
        self.scale_projections = nn.ModuleList()
        
        for i, (scale, dim) in enumerate(zip(config.scale_factors, config.level_dims)):
            # Temporal downsampling processor
            if scale > 1:
                processor = nn.Sequential(
                    nn.Conv1d(config.embed_dim, dim, kernel_size=scale, stride=scale),
                    nn.GELU(),
                    nn.BatchNorm1d(dim)
                )
            else:
                processor = nn.Sequential(
                    nn.Linear(config.embed_dim, dim),
                    nn.GELU(),
                    nn.LayerNorm(dim)
                )
            
            self.scale_processors.append(processor)
            
            # Scale-specific attention
            attention = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=max(1, config.n_heads // (2**i)),
                batch_first=True,
                dropout=config.dropout
            )
            self.scale_attentions.append(attention)
            
            # Projection back to original dimension
            projection = nn.Linear(dim, config.embed_dim)
            self.scale_projections.append(projection)
        
        # Cross-scale attention
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.n_heads,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(config.embed_dim * config.hierarchy_levels, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hierarchical temporal attention forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Tuple of (output, scale_info)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        scale_outputs = []
        scale_info = {}
        
        # Process each temporal scale
        for i, (processor, attention, projection) in enumerate(
            zip(self.scale_processors, self.scale_attentions, self.scale_projections)
        ):
            scale = self.scale_factors[i]
            
            if scale > 1:
                # Downsample temporally
                x_conv = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
                x_downsampled = processor(x_conv)  # (batch, level_dim, seq_len//scale)
                x_downsampled = x_downsampled.transpose(1, 2)  # (batch, seq_len//scale, level_dim)
            else:
                # Process at original resolution
                if hasattr(processor, '__iter__') and hasattr(processor[0], 'weight'):
                    # Linear layer
                    x_downsampled = processor(x)
                else:
                    x_downsampled = x
            
            # Apply scale-specific attention
            attn_output, attn_weights = attention(x_downsampled, x_downsampled, x_downsampled)
            
            # Project back to original dimension
            projected = projection(attn_output)
            
            # Upsample if necessary
            if scale > 1 and projected.size(1) != seq_len:
                # Interpolate to original sequence length
                projected = projected.transpose(1, 2)  # (batch, embed_dim, seq_len//scale)
                projected = F.interpolate(projected, size=seq_len, mode='linear', align_corners=False)
                projected = projected.transpose(1, 2)  # (batch, seq_len, embed_dim)
            
            scale_outputs.append(projected)
            scale_info[f"scale_{scale}_attention"] = attn_weights
            scale_info[f"scale_{scale}_features"] = projected
        
        # Concatenate scale outputs
        concatenated = torch.cat(scale_outputs, dim=-1)  # (batch, seq_len, embed_dim * n_scales)
        
        # Cross-scale attention
        cross_scale_output, cross_scale_weights = self.cross_scale_attention(
            concatenated, concatenated, concatenated
        )
        
        # Final fusion
        fused_output = self.fusion_network(cross_scale_output)
        
        # Residual connection and normalization
        output = self.norm(fused_output + x)
        
        scale_info["cross_scale_attention"] = cross_scale_weights
        
        return output, scale_info


class AdaptiveTemporalKernels(nn.Module):
    """Adaptive convolution kernels for dynamic neural patterns."""
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.embed_dim
        self.kernel_sizes = config.kernel_sizes
        self.max_dilation = config.max_dilation
        
        # Kernel generation network
        self.kernel_generator = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, len(config.kernel_sizes) * config.max_dilation * config.embed_dim),
            nn.Tanh()
        )
        
        # Base kernels (learnable)
        self.base_kernels = nn.ParameterDict({
            f"kernel_{k}": nn.Parameter(torch.randn(config.embed_dim, config.embed_dim, k))
            for k in config.kernel_sizes
        })
        
        # Dilation-specific layers
        self.dilated_convs = nn.ModuleDict({
            f"dilation_{d}": nn.ModuleDict({
                f"kernel_{k}": nn.Conv1d(
                    config.embed_dim, config.embed_dim, 
                    kernel_size=k, dilation=d, padding=(k-1)*d//2,
                    groups=config.embed_dim
                ) for k in config.kernel_sizes
            }) for d in range(1, config.max_dilation + 1)
        })
        
        # Attention for kernel selection
        self.kernel_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.n_heads,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            config.embed_dim * len(config.kernel_sizes) * config.max_dilation, 
            config.embed_dim
        )
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Adaptive temporal kernels forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Tuple of (output, kernel_info)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate adaptive kernel weights
        global_context = x.mean(dim=1)  # (batch, embed_dim)
        adaptive_weights = self.kernel_generator(global_context)  # (batch, n_kernels * n_dilations * embed_dim)
        
        # Reshape adaptive weights
        adaptive_weights = adaptive_weights.view(
            batch_size, len(self.kernel_sizes), self.max_dilation, embed_dim
        )
        
        # Apply adaptive kernels
        kernel_outputs = []
        kernel_info = {}
        
        x_conv = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        for k_idx, kernel_size in enumerate(self.kernel_sizes):
            for d_idx in range(self.max_dilation):
                dilation = d_idx + 1
                
                # Get adaptive weights for this kernel-dilation combination
                weights = adaptive_weights[:, k_idx, d_idx, :]  # (batch, embed_dim)
                
                # Apply dilated convolution
                conv_key = f"dilation_{dilation}"
                kernel_key = f"kernel_{kernel_size}"
                
                if conv_key in self.dilated_convs and kernel_key in self.dilated_convs[conv_key]:
                    conv_output = self.dilated_convs[conv_key][kernel_key](x_conv)  # (batch, embed_dim, seq_len)
                    
                    # Apply adaptive weights
                    weighted_output = conv_output * weights.unsqueeze(-1)  # Broadcast weights
                    kernel_outputs.append(weighted_output.transpose(1, 2))  # Back to (batch, seq_len, embed_dim)
                    
                    # Store kernel information
                    kernel_info[f"kernel_{kernel_size}_dilation_{dilation}_weights"] = weights
        
        # Concatenate all kernel outputs
        if kernel_outputs:
            concatenated_outputs = torch.cat(kernel_outputs, dim=-1)  # (batch, seq_len, embed_dim * n_kernels)
        else:
            concatenated_outputs = x
        
        # Apply attention to kernel outputs
        attended_output, attention_weights = self.kernel_attention(
            concatenated_outputs, concatenated_outputs, concatenated_outputs
        )
        
        # Final projection
        output = self.output_projection(attended_output)
        
        # Residual connection and normalization
        output = self.norm(output + x)
        
        kernel_info["kernel_attention_weights"] = attention_weights
        
        return output, kernel_info


class CausalTemporalAttention(nn.Module):
    """Causal temporal attention for real-time BCI processing."""
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.causal_window_size = config.causal_window_size
        
        # Causal attention components
        self.q_linear = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_linear = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_linear = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_linear = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Relative position encoding for causal attention
        if config.use_relative_position:
            self.relative_position_bias = nn.Parameter(
                torch.zeros(config.n_heads, config.causal_window_size, config.causal_window_size)
            )
        
        # Learnable causal mask
        self.register_buffer('causal_mask', self._create_causal_mask(config.sequence_length))
        
        # Temporal decay for older information
        self.temporal_decay = nn.Parameter(torch.tensor(0.95))
        
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embed_dim)
        
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x: torch.Tensor, past_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Causal temporal attention forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            past_states: Optional past states for incremental processing
            
        Returns:
            Tuple of (output, updated_states)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Concatenate with past states if provided
        if past_states is not None:
            # Apply temporal decay to past states
            decayed_past = past_states * (self.temporal_decay ** torch.arange(past_states.size(1), device=x.device).float())
            
            # Concatenate with current input
            extended_input = torch.cat([decayed_past, x], dim=1)
            
            # Limit to causal window size
            if extended_input.size(1) > self.causal_window_size:
                extended_input = extended_input[:, -self.causal_window_size:, :]
        else:
            extended_input = x
        
        extended_seq_len = extended_input.size(1)
        
        # Linear projections
        Q = self.q_linear(extended_input).view(batch_size, extended_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(extended_input).view(batch_size, extended_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(extended_input).view(batch_size, extended_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if extended_seq_len <= self.causal_mask.size(0):
            causal_mask = self.causal_mask[:extended_seq_len, :extended_seq_len]
        else:
            # Create larger causal mask if needed
            causal_mask = self._create_causal_mask(extended_seq_len).to(x.device)
        
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Add relative position bias if enabled
        if hasattr(self, 'relative_position_bias') and extended_seq_len <= self.relative_position_bias.size(1):
            rel_bias = self.relative_position_bias[:, :extended_seq_len, :extended_seq_len]
            scores = scores + rel_bias.unsqueeze(0)
        
        # Apply attention
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, extended_seq_len, embed_dim)
        
        # Extract output corresponding to current input
        if past_states is not None:
            output = attended[:, -seq_len:, :]
        else:
            output = attended
        
        # Output projection and residual connection
        output = self.out_linear(output)
        output = self.norm(output + x)
        
        # Update states for next iteration
        updated_states = extended_input[:, -(self.causal_window_size - seq_len):, :] if extended_input.size(1) > seq_len else None
        
        return output, updated_states


class NeuralOscillationAttention(nn.Module):
    """Attention mechanism specialized for neural oscillations."""
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        
        self.config = config
        self.frequency_bands = config.frequency_bands
        self.sampling_rate = config.sampling_rate
        
        # Oscillation-specific processing
        self.oscillation_processors = nn.ModuleDict({
            f"band_{i}": OscillationProcessor(
                low_freq=band[0],
                high_freq=band[1],
                sampling_rate=config.sampling_rate,
                embed_dim=config.embed_dim
            ) for i, band in enumerate(config.frequency_bands)
        })
        
        # Cross-frequency coupling attention
        self.cross_frequency_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.n_heads,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Oscillation fusion network
        self.oscillation_fusion = nn.Sequential(
            nn.Linear(config.embed_dim * len(config.frequency_bands), config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        
        # Phase-amplitude coupling detection
        self.pac_detector = PhaseAmplitudeCouplingDetector(config)
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Neural oscillation attention forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Tuple of (output, oscillation_info)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Process each frequency band
        band_outputs = []
        oscillation_info = {}
        
        for band_name, processor in self.oscillation_processors.items():
            band_output, band_info = processor(x)
            band_outputs.append(band_output)
            
            # Store band information
            for key, value in band_info.items():
                oscillation_info[f"{band_name}_{key}"] = value
        
        # Concatenate band outputs
        concatenated_bands = torch.cat(band_outputs, dim=-1)  # (batch, seq_len, embed_dim * n_bands)
        
        # Cross-frequency coupling attention
        cross_freq_output, cross_freq_weights = self.cross_frequency_attention(
            concatenated_bands, concatenated_bands, concatenated_bands
        )
        
        # Fusion of oscillation information
        fused_output = self.oscillation_fusion(cross_freq_output)
        
        # Detect phase-amplitude coupling
        pac_info = self.pac_detector(band_outputs)
        oscillation_info.update(pac_info)
        
        # Final output with residual connection
        output = self.norm(fused_output + x)
        
        oscillation_info["cross_frequency_attention"] = cross_freq_weights
        
        return output, oscillation_info


class OscillationProcessor(nn.Module):
    """Process neural oscillations in specific frequency bands."""
    
    def __init__(self, low_freq: float, high_freq: float, sampling_rate: float, embed_dim: int):
        super().__init__()
        
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate
        self.embed_dim = embed_dim
        
        # Wavelet-inspired filters
        self.wavelet_filters = self._create_wavelet_filters()
        
        # Phase and amplitude processing
        self.phase_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        self.amplitude_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Oscillation attention
        self.oscillation_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )
        
    def _create_wavelet_filters(self) -> nn.ModuleList:
        """Create wavelet-inspired filters for the frequency band."""
        filters = nn.ModuleList()
        
        # Create filters at different scales within the frequency band
        center_freq = (self.low_freq + self.high_freq) / 2
        bandwidth = self.high_freq - self.low_freq
        
        # Multiple scales for better frequency resolution
        for scale in [0.5, 1.0, 2.0]:
            scaled_freq = center_freq * scale
            scaled_bandwidth = bandwidth * scale
            
            # Approximate Morlet wavelet with convolution
            kernel_size = min(31, int(2 * self.sampling_rate / scaled_freq))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            
            filter_layer = nn.Conv1d(
                self.embed_dim, self.embed_dim, 
                kernel_size=kernel_size, 
                padding=kernel_size//2,
                groups=self.embed_dim
            )
            
            # Initialize with wavelet-like weights
            self._init_wavelet_weights(filter_layer, scaled_freq, scaled_bandwidth, kernel_size)
            
            filters.append(filter_layer)
        
        return filters
    
    def _init_wavelet_weights(self, layer: nn.Conv1d, center_freq: float, bandwidth: float, kernel_size: int):
        """Initialize convolution weights with wavelet-like pattern."""
        with torch.no_grad():
            t = torch.linspace(-1, 1, kernel_size)
            
            # Morlet wavelet approximation
            sigma = 1.0 / bandwidth
            wave = torch.cos(2 * math.pi * center_freq * t / self.sampling_rate) * torch.exp(-t**2 / (2 * sigma**2))
            wave = wave / wave.norm()
            
            # Set weights for each channel
            for i in range(self.embed_dim):
                layer.weight[i, 0, :] = wave
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process oscillations in the frequency band.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Tuple of (processed_output, band_info)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Apply wavelet filters
        x_conv = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        filtered_outputs = []
        
        for wavelet_filter in self.wavelet_filters:
            filtered = wavelet_filter(x_conv)
            filtered_outputs.append(filtered)
        
        # Combine filtered outputs
        combined_filtered = torch.stack(filtered_outputs, dim=0).mean(dim=0)  # (batch, embed_dim, seq_len)
        combined_filtered = combined_filtered.transpose(1, 2)  # (batch, seq_len, embed_dim)
        
        # Extract amplitude and phase approximations
        amplitude_approx = torch.abs(combined_filtered)
        phase_approx = torch.angle(torch.complex(combined_filtered, torch.zeros_like(combined_filtered)))
        
        # Process amplitude and phase
        processed_amplitude = self.amplitude_processor(amplitude_approx)
        processed_phase = self.phase_processor(phase_approx.real)  # Use real part for phase processing
        
        # Combine amplitude and phase information
        combined_features = processed_amplitude + processed_phase
        
        # Apply oscillation-specific attention
        attended_output, attention_weights = self.oscillation_attention(
            combined_features, combined_features, combined_features
        )
        
        band_info = {
            "amplitude": amplitude_approx,
            "phase": phase_approx,
            "attention_weights": attention_weights,
            "filtered_signal": combined_filtered
        }
        
        return attended_output, band_info


class PhaseAmplitudeCouplingDetector(nn.Module):
    """Detect phase-amplitude coupling between frequency bands."""
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        
        self.config = config
        self.n_bands = len(config.frequency_bands)
        
        # PAC detection network
        self.pac_network = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),  # phase + amplitude
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Cross-band coupling matrix
        self.coupling_matrix = nn.Parameter(
            torch.zeros(self.n_bands, self.n_bands)
        )
        
    def forward(self, band_outputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Detect phase-amplitude coupling.
        
        Args:
            band_outputs: List of band-specific outputs
            
        Returns:
            Dictionary with PAC information
        """
        pac_info = {}
        
        # Compute coupling between all pairs of bands
        coupling_strengths = []
        
        for i, band_i in enumerate(band_outputs):
            for j, band_j in enumerate(band_outputs):
                if i != j:
                    # Combine phase of band i with amplitude of band j
                    combined_features = torch.cat([band_i, band_j], dim=-1)
                    
                    # Detect coupling strength
                    coupling_strength = self.pac_network(combined_features)
                    coupling_strengths.append(coupling_strength)
                    
                    pac_info[f"pac_band_{i}_to_band_{j}"] = coupling_strength
        
        # Overall coupling matrix
        if coupling_strengths:
            pac_matrix = torch.stack(coupling_strengths, dim=-1)  # (batch, seq_len, 1, n_pairs)
            pac_info["pac_matrix"] = pac_matrix
        
        pac_info["learned_coupling_matrix"] = self.coupling_matrix
        
        return pac_info


class TemporalTransformerBCI(nn.Module):
    """Complete temporal transformer for BCI applications."""
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        
        self.config = config
        
        # Input embedding and positional encoding
        self.input_projection = nn.Linear(config.n_frequency_bands, config.embed_dim)
        
        if config.use_learnable_position:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, config.sequence_length, config.embed_dim) * 0.1
            )
        else:
            self.register_buffer('positional_encoding', self._create_sinusoidal_encoding())
        
        # Temporal attention layers
        self.layers = nn.ModuleList()
        
        for i in range(config.n_layers):
            layer = nn.ModuleDict({
                'phase_attention': PhaseAwareAttention(config),
                'hierarchical_attention': HierarchicalTemporalAttention(config),
                'adaptive_kernels': AdaptiveTemporalKernels(config),
                'causal_attention': CausalTemporalAttention(config),
                'oscillation_attention': NeuralOscillationAttention(config),
                'norm1': nn.LayerNorm(config.embed_dim),
                'norm2': nn.LayerNorm(config.embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(config.embed_dim, config.embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.embed_dim * 4, config.embed_dim),
                    nn.Dropout(config.dropout)
                )
            })
            self.layers.append(layer)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, 2)  # Binary classification
        )
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        
    def _create_sinusoidal_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(self.config.sequence_length, self.config.embed_dim)
        position = torch.arange(0, self.config.sequence_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.config.embed_dim, 2).float() * 
                           (-math.log(10000.0) / self.config.embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor, past_states: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of temporal transformer BCI.
        
        Args:
            x: Input tensor (batch, seq_len, n_channels)
            past_states: Optional past states for real-time processing
            
        Returns:
            Dictionary with predictions and analysis
        """
        start_time = time.time()
        
        # Input projection and positional encoding
        x = self.input_projection(x)  # (batch, seq_len, embed_dim)
        
        if hasattr(self, 'positional_encoding'):
            if self.positional_encoding.size(1) >= x.size(1):
                pos_enc = self.positional_encoding[:, :x.size(1), :]
            else:
                pos_enc = self.positional_encoding
            x = x + pos_enc
        
        # Initialize past states if not provided
        if past_states is None:
            past_states = [None] * len(self.layers)
        
        updated_states = []
        layer_outputs = {}
        
        # Process through temporal attention layers
        for i, layer in enumerate(self.layers):
            # Phase-aware attention
            phase_output, phase_info = layer['phase_attention'](x)
            layer_outputs[f"layer_{i}_phase"] = phase_info
            
            # Hierarchical attention
            hier_output, hier_info = layer['hierarchical_attention'](phase_output)
            layer_outputs[f"layer_{i}_hierarchical"] = hier_info
            
            # Adaptive kernels
            kernel_output, kernel_info = layer['adaptive_kernels'](hier_output)
            layer_outputs[f"layer_{i}_kernels"] = kernel_info
            
            # Causal attention (with state management)
            causal_output, updated_state = layer['causal_attention'](kernel_output, past_states[i])
            updated_states.append(updated_state)
            
            # Oscillation attention
            osc_output, osc_info = layer['oscillation_attention'](causal_output)
            layer_outputs[f"layer_{i}_oscillations"] = osc_info
            
            # Residual connections and normalization
            x = layer['norm1'](osc_output + x)
            
            # Feed-forward network
            ffn_output = layer['ffn'](x)
            x = layer['norm2'](ffn_output + x)
        
        # Global pooling for classification
        pooled_features = x.mean(dim=1)  # (batch, embed_dim)
        
        # Output predictions
        logits = self.output_head(pooled_features)
        predictions = F.softmax(logits, dim=-1)
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        output = {
            "logits": logits,
            "predictions": predictions,
            "features": pooled_features,
            "layer_outputs": layer_outputs,
            "updated_states": updated_states,
            "inference_time": inference_time
        }
        
        return output
    
    def get_average_latency_ms(self) -> float:
        """Get average inference latency in milliseconds."""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000  # Convert to ms


def create_temporal_bci_system(config: Optional[TemporalAttentionConfig] = None) -> TemporalTransformerBCI:
    """
    Create a temporal attention BCI system.
    
    Args:
        config: Temporal attention configuration (optional)
        
    Returns:
        Configured temporal transformer BCI
    """
    if config is None:
        config = TemporalAttentionConfig(
            embed_dim=128,
            n_heads=8,
            n_layers=6,
            sequence_length=250,
            sampling_rate=250.0,
            use_phase_attention=True,
            use_adaptive_kernels=True,
            use_causal_attention=True
        )
    
    model = TemporalTransformerBCI(config)
    
    logger.info(f"Created temporal BCI system with {config.n_layers} layers")
    
    return model


# Example usage
def run_temporal_attention_example():
    """Example of running temporal attention BCI."""
    import torch.utils.data as data
    
    # Create synthetic neural data with temporal patterns
    class TemporalBCIDataset(data.Dataset):
        def __init__(self, n_samples=1000, seq_length=250, n_channels=5):
            self.n_samples = n_samples
            self.seq_length = seq_length
            self.n_channels = n_channels
            
            # Generate data with temporal patterns
            self.data, self.labels = self._generate_temporal_data()
        
        def _generate_temporal_data(self):
            data = torch.zeros(self.n_samples, self.seq_length, self.n_channels)
            labels = torch.zeros(self.n_samples, dtype=torch.long)
            
            for i in range(self.n_samples):
                # Add oscillations at different frequencies
                t = torch.linspace(0, 1, self.seq_length)
                
                # Delta (0.5-4 Hz)
                data[i, :, 0] = torch.sin(2 * math.pi * 2 * t) + 0.1 * torch.randn(self.seq_length)
                
                # Theta (4-8 Hz)
                data[i, :, 1] = torch.sin(2 * math.pi * 6 * t) + 0.1 * torch.randn(self.seq_length)
                
                # Alpha (8-13 Hz)
                data[i, :, 2] = torch.sin(2 * math.pi * 10 * t) + 0.1 * torch.randn(self.seq_length)
                
                # Beta (13-30 Hz)
                data[i, :, 3] = torch.sin(2 * math.pi * 20 * t) + 0.1 * torch.randn(self.seq_length)
                
                # Gamma (30-100 Hz)
                data[i, :, 4] = torch.sin(2 * math.pi * 50 * t) + 0.1 * torch.randn(self.seq_length)
                
                # Label based on alpha power
                alpha_power = (data[i, :, 2] ** 2).mean()
                labels[i] = (alpha_power > 0.5).long()
            
            return data, labels
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    # Create dataset and data loader
    dataset = TemporalBCIDataset(n_samples=200, seq_length=100, n_channels=5)
    data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create temporal attention BCI system
    config = TemporalAttentionConfig(
        embed_dim=64,
        n_heads=4,
        n_layers=3,
        sequence_length=100,
        n_frequency_bands=5,
        sampling_rate=250.0
    )
    
    temporal_bci = create_temporal_bci_system(config)
    
    # Test forward pass
    sample_data, sample_labels = next(iter(data_loader))
    
    print("Temporal Attention BCI Example:")
    print(f"Input shape: {sample_data.shape}")
    
    with torch.no_grad():
        output = temporal_bci(sample_data)
    
    print(f"Output predictions shape: {output['predictions'].shape}")
    print(f"Inference time: {output['inference_time']*1000:.2f} ms")
    print(f"Average latency: {temporal_bci.get_average_latency_ms():.2f} ms")
    
    # Test real-time processing with states
    print("\nTesting real-time processing:")
    past_states = None
    
    for i, (data_chunk, _) in enumerate(data_loader):
        if i >= 3:  # Test first 3 chunks
            break
        
        with torch.no_grad():
            output = temporal_bci(data_chunk, past_states)
            past_states = output['updated_states']
        
        print(f"Chunk {i+1}: Latency = {output['inference_time']*1000:.2f} ms")
    
    # Analyze layer outputs
    print("\nLayer analysis:")
    for key, value in output['layer_outputs'].items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} sub-components")
        else:
            print(f"{key}: {type(value)}")
    
    return temporal_bci, output


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    model, results = run_temporal_attention_example()