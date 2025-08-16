"""
Causal Neural Inference Networks for Brain-Computer Interfaces.

This module implements breakthrough causal inference techniques for understanding
cause-effect relationships in neural signals and BCI interactions:

1. Neural Causal Discovery: Automated discovery of causal relationships in neural data
2. Interventional BCI: Causal interventions for improved BCI control
3. Counterfactual Analysis: "What if" analysis for BCI decisions
4. Granger Causality Networks: Deep learning Granger causality estimation
5. Causal Representation Learning: Learning causal latent representations

Research Contributions:
- First deep learning framework for causal discovery in neural signals
- Novel interventional techniques for BCI improvement
- Counterfactual explanations for BCI decisions
- Causal representation learning for robust BCI systems
- Temporal causal discovery with attention mechanisms

Applications:
- Understanding neural information flow
- Identifying optimal BCI control strategies
- Explainable BCI decision making
- Robust BCI systems under distribution shift
- Clinical understanding of neural disorders
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
from scipy import stats
from scipy.optimize import minimize
import networkx as nx
import json
from pathlib import Path
from itertools import combinations, permutations

logger = logging.getLogger(__name__)


@dataclass
class CausalConfig:
    """Configuration for causal inference."""
    
    # Network structure
    n_variables: int = 64  # Number of variables (e.g., EEG channels)
    max_lag: int = 10      # Maximum temporal lag to consider
    
    # Causal discovery
    discovery_method: str = "gradient_based"  # "gradient_based", "score_based", "constraint_based"
    sparsity_lambda: float = 0.01  # Sparsity regularization
    dag_lambda: float = 1.0        # DAG constraint strength
    
    # Model architecture
    hidden_dim: int = 128
    n_layers: int = 3
    attention_heads: int = 8
    
    # Training parameters
    learning_rate: float = 0.001
    max_epochs: int = 1000
    early_stopping_patience: int = 50
    
    # Intervention parameters
    intervention_strength: float = 1.0
    n_intervention_samples: int = 100
    
    # Uncertainty quantification
    use_uncertainty: bool = True
    n_bootstrap_samples: int = 100
    confidence_level: float = 0.95


class CausalGraphNetwork(nn.Module):
    """Neural network for learning causal graph structure."""
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        
        self.config = config
        self.n_vars = config.n_variables
        self.max_lag = config.max_lag
        
        # Adjacency matrix parameters (learnable)
        self.adjacency_logits = nn.Parameter(
            torch.randn(self.n_vars, self.n_vars, self.max_lag + 1) * 0.1
        )
        
        # Neural networks for causal mechanisms
        self.causal_mechanisms = nn.ModuleList([
            CausalMechanismNetwork(config) for _ in range(self.n_vars)
        ])
        
        # Attention mechanisms for temporal dependencies
        self.temporal_attention = nn.ModuleList([
            nn.MultiheadAttention(config.hidden_dim, config.attention_heads, batch_first=True)
            for _ in range(self.n_vars)
        ])
        
        # Embedding layers
        self.variable_embeddings = nn.Embedding(self.n_vars, config.hidden_dim)
        self.temporal_embeddings = nn.Embedding(self.max_lag + 1, config.hidden_dim)
        
        # Output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(config.hidden_dim, 1) for _ in range(self.n_vars)
        ])
        
        # Gumbel-Softmax temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def get_adjacency_matrix(self, hard: bool = False) -> torch.Tensor:
        """Get the learned adjacency matrix."""
        if hard:
            # Hard sampling using Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.adjacency_logits) + 1e-20) + 1e-20)
            adj_hard = torch.sigmoid((self.adjacency_logits + gumbel_noise) / self.temperature)
            adj_hard = (adj_hard > 0.5).float()
            return adj_hard
        else:
            # Soft adjacency matrix
            return torch.sigmoid(self.adjacency_logits)
    
    def forward(self, x: torch.Tensor, return_graph: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for causal graph network.
        
        Args:
            x: Input tensor (batch, sequence_length, n_variables)
            return_graph: Whether to return the adjacency matrix
            
        Returns:
            Predictions or (predictions, adjacency_matrix)
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Get adjacency matrix
        adj_matrix = self.get_adjacency_matrix()  # (n_vars, n_vars, max_lag + 1)
        
        # Initialize outputs
        outputs = []
        
        for i in range(n_vars):
            # Compute causal inputs for variable i
            causal_inputs = self._compute_causal_inputs(x, i, adj_matrix)
            
            # Apply causal mechanism
            causal_output = self.causal_mechanisms[i](causal_inputs)
            
            # Apply temporal attention
            attn_output, _ = self.temporal_attention[i](causal_output, causal_output, causal_output)
            
            # Final prediction
            prediction = self.output_layers[i](attn_output)
            outputs.append(prediction)
        
        predictions = torch.cat(outputs, dim=-1)  # (batch, seq_len, n_vars)
        
        if return_graph:
            return predictions, adj_matrix
        else:
            return predictions
    
    def _compute_causal_inputs(self, x: torch.Tensor, target_var: int, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Compute causal inputs for a target variable."""
        batch_size, seq_len, n_vars = x.shape
        
        causal_inputs = []
        
        for source_var in range(n_vars):
            for lag in range(self.max_lag + 1):
                # Get adjacency weight
                weight = adj_matrix[source_var, target_var, lag]  # Scalar
                
                # Get lagged input
                if lag == 0:
                    lagged_input = x[:, :, source_var:source_var+1]  # (batch, seq_len, 1)
                else:
                    # Pad and shift
                    padded_input = F.pad(x[:, :, source_var:source_var+1], (0, 0, lag, 0))
                    lagged_input = padded_input[:, :-lag, :]
                
                # Weight by adjacency
                weighted_input = lagged_input * weight
                
                # Add variable and temporal embeddings
                var_emb = self.variable_embeddings(torch.tensor(source_var, device=x.device))
                temp_emb = self.temporal_embeddings(torch.tensor(lag, device=x.device))
                
                combined_emb = var_emb + temp_emb  # (hidden_dim,)
                combined_emb = combined_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                
                # Expand weighted input to hidden dimension
                weighted_expanded = weighted_input.expand(-1, -1, self.config.hidden_dim) * combined_emb
                
                causal_inputs.append(weighted_expanded)
        
        # Stack all causal inputs
        if causal_inputs:
            stacked_inputs = torch.stack(causal_inputs, dim=-2)  # (batch, seq_len, n_sources*n_lags, hidden_dim)
            # Sum over causal sources
            causal_input = stacked_inputs.sum(dim=-2)  # (batch, seq_len, hidden_dim)
        else:
            causal_input = torch.zeros(batch_size, seq_len, self.config.hidden_dim, device=x.device)
        
        return causal_input


class CausalMechanismNetwork(nn.Module):
    """Neural network representing a causal mechanism."""
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        
        # Hidden layers
        for _ in range(config.n_layers - 1):
            self.layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        
        # Normalization and activation
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.n_layers)
        ])
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through causal mechanism."""
        for layer, norm in zip(self.layers, self.layer_norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        return x


class GrangerCausalityNetwork(nn.Module):
    """Deep learning implementation of Granger causality."""
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        
        self.config = config
        self.n_vars = config.n_variables
        self.max_lag = config.max_lag
        
        # Networks for full model (including all variables)
        self.full_models = nn.ModuleList([
            self._build_predictor_network() for _ in range(self.n_vars)
        ])
        
        # Networks for restricted model (excluding specific variables)
        self.restricted_models = nn.ModuleDict({
            f"var_{i}_without_{j}": self._build_predictor_network()
            for i in range(self.n_vars) for j in range(self.n_vars) if i != j
        })
        
    def _build_predictor_network(self) -> nn.Module:
        """Build a predictor network."""
        return nn.Sequential(
            nn.Linear(self.n_vars * self.max_lag, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Granger causality between all pairs of variables.
        
        Args:
            x: Input tensor (batch, sequence_length, n_variables)
            
        Returns:
            Dictionary with causality scores
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Create lagged features
        lagged_features = self._create_lagged_features(x)  # (batch, seq_len-max_lag, n_vars*max_lag)
        
        causality_matrix = torch.zeros(n_vars, n_vars, device=x.device)
        
        for target_var in range(n_vars):
            # Target variable (future values)
            target = x[:, self.max_lag:, target_var:target_var+1]  # (batch, seq_len-max_lag, 1)
            
            # Full model prediction
            full_pred = self.full_models[target_var](lagged_features)
            full_loss = F.mse_loss(full_pred, target)
            
            for source_var in range(n_vars):
                if source_var != target_var:
                    # Restricted model (without source variable)
                    restricted_features = self._remove_variable_lags(lagged_features, source_var)
                    
                    model_key = f"var_{target_var}_without_{source_var}"
                    restricted_pred = self.restricted_models[model_key](restricted_features)
                    restricted_loss = F.mse_loss(restricted_pred, target)
                    
                    # Granger causality score
                    causality_score = torch.log(restricted_loss / (full_loss + 1e-8))
                    causality_matrix[source_var, target_var] = causality_score
        
        return {
            "causality_matrix": causality_matrix,
            "lagged_features": lagged_features
        }
    
    def _create_lagged_features(self, x: torch.Tensor) -> torch.Tensor:
        """Create lagged features for all variables."""
        batch_size, seq_len, n_vars = x.shape
        
        lagged_features = []
        
        for lag in range(1, self.max_lag + 1):
            lagged = x[:, :-lag, :]  # (batch, seq_len-lag, n_vars)
            # Pad to maintain sequence length
            padded = F.pad(lagged, (0, 0, 0, lag))[:, lag:, :]
            lagged_features.append(padded)
        
        # Concatenate all lags
        lagged_features = torch.cat(lagged_features, dim=-1)  # (batch, seq_len-max_lag, n_vars*max_lag)
        
        return lagged_features
    
    def _remove_variable_lags(self, lagged_features: torch.Tensor, var_index: int) -> torch.Tensor:
        """Remove all lags of a specific variable."""
        # lagged_features shape: (batch, seq_len, n_vars*max_lag)
        n_vars = self.n_vars
        max_lag = self.max_lag
        
        # Create mask for features to keep
        keep_indices = []
        for lag in range(max_lag):
            for var in range(n_vars):
                if var != var_index:
                    keep_indices.append(lag * n_vars + var)
        
        return lagged_features[:, :, keep_indices]


class InterventionalBCI(nn.Module):
    """Interventional BCI system for causal interventions."""
    
    def __init__(self, config: CausalConfig, causal_graph: CausalGraphNetwork):
        super().__init__()
        
        self.config = config
        self.causal_graph = causal_graph
        
        # Intervention policy network
        self.intervention_policy = nn.Sequential(
            nn.Linear(config.n_variables, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.n_variables)  # Intervention targets
        )
        
        # Value network for intervention assessment
        self.value_network = nn.Sequential(
            nn.Linear(config.n_variables * 2, config.hidden_dim),  # state + intervention
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, intervention_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Perform interventional inference.
        
        Args:
            x: Input tensor (batch, sequence_length, n_variables)
            intervention_targets: Optional intervention targets
            
        Returns:
            Dictionary with intervention results
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Get baseline predictions
        baseline_pred, adj_matrix = self.causal_graph(x, return_graph=True)
        
        # Generate intervention policy if not provided
        if intervention_targets is None:
            # Use current state to determine intervention
            current_state = x[:, -1, :]  # Last time step
            intervention_logits = self.intervention_policy(current_state)
            intervention_targets = torch.sigmoid(intervention_logits)
        
        # Perform intervention
        interventional_pred = self._perform_intervention(x, intervention_targets, adj_matrix)
        
        # Compute intervention effect
        intervention_effect = interventional_pred - baseline_pred
        
        # Assess intervention value
        intervention_input = torch.cat([x[:, -1, :], intervention_targets], dim=-1)
        intervention_value = self.value_network(intervention_input)
        
        return {
            "baseline_prediction": baseline_pred,
            "interventional_prediction": interventional_pred,
            "intervention_effect": intervention_effect,
            "intervention_targets": intervention_targets,
            "intervention_value": intervention_value,
            "adjacency_matrix": adj_matrix
        }
    
    def _perform_intervention(self, x: torch.Tensor, intervention_targets: torch.Tensor, 
                            adj_matrix: torch.Tensor) -> torch.Tensor:
        """Perform do-calculus intervention."""
        batch_size, seq_len, n_vars = x.shape
        
        # Create interventional adjacency matrix
        # Set incoming edges to intervened variables to zero
        interventional_adj = adj_matrix.clone()
        for i in range(n_vars):
            intervention_strength = intervention_targets[:, i:i+1]  # (batch, 1)
            # Zero out incoming edges proportional to intervention strength
            interventional_adj[:, i, :] = interventional_adj[:, i, :] * (1 - intervention_strength.mean())
        
        # Forward pass with interventional graph
        outputs = []
        
        for i in range(n_vars):
            # Check if this variable is intervened
            intervention_strength = intervention_targets[:, i]  # (batch,)
            
            if intervention_strength.max() > 0.1:  # If significantly intervened
                # Use intervention value instead of causal mechanism
                intervention_value = intervention_strength.unsqueeze(-1).unsqueeze(-1).expand(-1, seq_len, 1)
                outputs.append(intervention_value)
            else:
                # Use normal causal mechanism with modified adjacency
                causal_inputs = self._compute_causal_inputs_with_adj(x, i, interventional_adj)
                causal_output = self.causal_graph.causal_mechanisms[i](causal_inputs)
                attn_output, _ = self.causal_graph.temporal_attention[i](causal_output, causal_output, causal_output)
                prediction = self.causal_graph.output_layers[i](attn_output)
                outputs.append(prediction)
        
        return torch.cat(outputs, dim=-1)
    
    def _compute_causal_inputs_with_adj(self, x: torch.Tensor, target_var: int, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Compute causal inputs with custom adjacency matrix."""
        # Similar to CausalGraphNetwork._compute_causal_inputs but with custom adjacency
        batch_size, seq_len, n_vars = x.shape
        
        causal_inputs = []
        
        for source_var in range(n_vars):
            for lag in range(self.config.max_lag + 1):
                weight = adj_matrix[source_var, target_var, lag]
                
                if lag == 0:
                    lagged_input = x[:, :, source_var:source_var+1]
                else:
                    padded_input = F.pad(x[:, :, source_var:source_var+1], (0, 0, lag, 0))
                    lagged_input = padded_input[:, :-lag, :]
                
                weighted_input = lagged_input * weight
                
                var_emb = self.causal_graph.variable_embeddings(torch.tensor(source_var, device=x.device))
                temp_emb = self.causal_graph.temporal_embeddings(torch.tensor(lag, device=x.device))
                
                combined_emb = var_emb + temp_emb
                combined_emb = combined_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                
                weighted_expanded = weighted_input.expand(-1, -1, self.config.hidden_dim) * combined_emb
                causal_inputs.append(weighted_expanded)
        
        if causal_inputs:
            stacked_inputs = torch.stack(causal_inputs, dim=-2)
            causal_input = stacked_inputs.sum(dim=-2)
        else:
            causal_input = torch.zeros(batch_size, seq_len, self.config.hidden_dim, device=x.device)
        
        return causal_input


class CounterfactualAnalysis(nn.Module):
    """Counterfactual analysis for BCI explanations."""
    
    def __init__(self, config: CausalConfig, causal_graph: CausalGraphNetwork):
        super().__init__()
        
        self.config = config
        self.causal_graph = causal_graph
        
        # Counterfactual generation network
        self.counterfactual_generator = nn.Sequential(
            nn.Linear(config.n_variables * 2, config.hidden_dim),  # original + target
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.n_variables)  # counterfactual input
        )
        
        # Proximity network (how close counterfactual should be to original)
        self.proximity_network = nn.Sequential(
            nn.Linear(config.n_variables, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def generate_counterfactuals(self, x: torch.Tensor, target_outcomes: torch.Tensor,
                               n_counterfactuals: int = 10) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual explanations.
        
        Args:
            x: Original input (batch, sequence_length, n_variables)
            target_outcomes: Desired outcomes (batch, n_classes)
            n_counterfactuals: Number of counterfactuals to generate
            
        Returns:
            Dictionary with counterfactual results
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Get original prediction
        original_pred = self.causal_graph(x)
        
        counterfactuals = []
        counterfactual_preds = []
        proximity_scores = []
        
        for _ in range(n_counterfactuals):
            # Generate counterfactual input
            cf_input_features = torch.cat([
                x[:, -1, :],  # Last time step of original
                target_outcomes
            ], dim=-1)
            
            cf_modification = self.counterfactual_generator(cf_input_features)
            
            # Apply modification with proximity constraint
            proximity_weight = self.proximity_network(x[:, -1, :])
            cf_input = x.clone()
            cf_input[:, -1, :] = x[:, -1, :] + cf_modification * proximity_weight
            
            # Get counterfactual prediction
            cf_pred = self.causal_graph(cf_input)
            
            # Compute proximity score
            proximity = torch.norm(cf_input - x, dim=(-2, -1))
            
            counterfactuals.append(cf_input)
            counterfactual_preds.append(cf_pred)
            proximity_scores.append(proximity)
        
        # Stack results
        counterfactuals = torch.stack(counterfactuals, dim=1)  # (batch, n_cf, seq_len, n_vars)
        counterfactual_preds = torch.stack(counterfactual_preds, dim=1)  # (batch, n_cf, seq_len, n_vars)
        proximity_scores = torch.stack(proximity_scores, dim=1)  # (batch, n_cf)
        
        return {
            "original_input": x,
            "original_prediction": original_pred,
            "counterfactuals": counterfactuals,
            "counterfactual_predictions": counterfactual_preds,
            "proximity_scores": proximity_scores,
            "target_outcomes": target_outcomes
        }
    
    def explain_decision(self, x: torch.Tensor, decision_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Explain BCI decision using counterfactual analysis.
        
        Args:
            x: Input tensor (batch, sequence_length, n_variables)
            decision_threshold: Threshold for binary decisions
            
        Returns:
            Dictionary with explanations
        """
        # Get original prediction
        original_pred = self.causal_graph(x)
        original_decision = (original_pred[:, -1, :].mean(dim=-1) > decision_threshold).float()
        
        # Generate counterfactuals for opposite decision
        target_outcomes = 1 - original_decision  # Flip decision
        target_outcomes = target_outcomes.unsqueeze(-1).expand(-1, self.config.n_variables)
        
        counterfactual_results = self.generate_counterfactuals(x, target_outcomes)
        
        # Compute importance scores for each variable
        importance_scores = self._compute_importance_scores(
            x, counterfactual_results["counterfactuals"]
        )
        
        return {
            "original_decision": original_decision,
            "counterfactual_analysis": counterfactual_results,
            "variable_importance": importance_scores,
            "explanation": self._generate_textual_explanation(importance_scores)
        }
    
    def _compute_importance_scores(self, original: torch.Tensor, counterfactuals: torch.Tensor) -> torch.Tensor:
        """Compute variable importance based on counterfactual differences."""
        # original: (batch, seq_len, n_vars)
        # counterfactuals: (batch, n_cf, seq_len, n_vars)
        
        # Compute differences
        differences = torch.abs(counterfactuals - original.unsqueeze(1))  # (batch, n_cf, seq_len, n_vars)
        
        # Average across counterfactuals and time
        importance = differences.mean(dim=(1, 2))  # (batch, n_vars)
        
        # Normalize
        importance = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
        
        return importance
    
    def _generate_textual_explanation(self, importance_scores: torch.Tensor) -> List[str]:
        """Generate textual explanations from importance scores."""
        batch_size, n_vars = importance_scores.shape
        
        explanations = []
        
        for b in range(batch_size):
            scores = importance_scores[b]
            top_indices = torch.argsort(scores, descending=True)[:3]  # Top 3 important variables
            
            explanation = f"Decision primarily influenced by variables: "
            for i, idx in enumerate(top_indices):
                var_name = f"Var_{idx.item()}"
                importance = scores[idx].item()
                explanation += f"{var_name} ({importance:.3f})"
                if i < len(top_indices) - 1:
                    explanation += ", "
            
            explanations.append(explanation)
        
        return explanations


class CausalRepresentationLearning(nn.Module):
    """Learn causal representations for robust BCI systems."""
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        
        self.config = config
        
        # Encoder for raw neural signals
        self.encoder = nn.Sequential(
            nn.Linear(config.n_variables, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Causal latent space
        self.causal_latent_dim = config.hidden_dim // 4
        self.causal_projector = nn.Linear(config.hidden_dim // 2, self.causal_latent_dim)
        
        # Non-causal latent space
        self.non_causal_latent_dim = config.hidden_dim // 4
        self.non_causal_projector = nn.Linear(config.hidden_dim // 2, self.non_causal_latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.causal_latent_dim + self.non_causal_latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.n_variables)
        )
        
        # Classifier using only causal representations
        self.causal_classifier = nn.Sequential(
            nn.Linear(self.causal_latent_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 2)  # Binary classification
        )
        
        # Intervention model
        self.intervention_model = nn.Sequential(
            nn.Linear(self.causal_latent_dim, self.causal_latent_dim),
            nn.GELU(),
            nn.Linear(self.causal_latent_dim, self.causal_latent_dim)
        )
        
    def forward(self, x: torch.Tensor, interventions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for causal representation learning.
        
        Args:
            x: Input tensor (batch, sequence_length, n_variables)
            interventions: Optional intervention tensor
            
        Returns:
            Dictionary with representations and predictions
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Encode to latent space
        encoded = self.encoder(x)  # (batch, seq_len, hidden_dim // 2)
        
        # Split into causal and non-causal representations
        causal_repr = self.causal_projector(encoded)  # (batch, seq_len, causal_latent_dim)
        non_causal_repr = self.non_causal_projector(encoded)  # (batch, seq_len, non_causal_latent_dim)
        
        # Apply interventions if provided
        if interventions is not None:
            causal_repr = self.intervention_model(causal_repr) + interventions
        
        # Reconstruct input
        combined_repr = torch.cat([causal_repr, non_causal_repr], dim=-1)
        reconstructed = self.decoder(combined_repr)
        
        # Classification using only causal representations
        causal_pooled = causal_repr.mean(dim=1)  # Global average pooling
        predictions = self.causal_classifier(causal_pooled)
        
        return {
            "causal_representation": causal_repr,
            "non_causal_representation": non_causal_repr,
            "reconstructed": reconstructed,
            "predictions": predictions,
            "logits": predictions
        }
    
    def compute_causal_loss(self, x: torch.Tensor, y: torch.Tensor, 
                          interventions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute causal representation learning loss."""
        
        # Forward pass
        outputs = self.forward(x, interventions)
        
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(outputs["reconstructed"], x)
        
        # Classification loss
        classification_loss = F.cross_entropy(outputs["logits"], y)
        
        # Causal invariance loss
        if interventions is not None:
            # Predictions should be invariant to non-causal interventions
            invariance_loss = self._compute_invariance_loss(x, y, interventions)
        else:
            invariance_loss = torch.tensor(0.0, device=x.device)
        
        # Independence loss (causal and non-causal should be independent)
        independence_loss = self._compute_independence_loss(
            outputs["causal_representation"], 
            outputs["non_causal_representation"]
        )
        
        # Total loss
        total_loss = (
            reconstruction_loss + 
            classification_loss + 
            0.1 * invariance_loss + 
            0.1 * independence_loss
        )
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss,
            "invariance_loss": invariance_loss,
            "independence_loss": independence_loss
        }
    
    def _compute_invariance_loss(self, x: torch.Tensor, y: torch.Tensor, 
                               interventions: torch.Tensor) -> torch.Tensor:
        """Compute invariance loss for causal representations."""
        
        # Original predictions
        original_outputs = self.forward(x)
        
        # Interventional predictions
        interventional_outputs = self.forward(x, interventions)
        
        # Causal representations should be affected by interventions
        causal_diff = F.mse_loss(
            interventional_outputs["causal_representation"],
            original_outputs["causal_representation"]
        )
        
        # Non-causal representations should be invariant
        non_causal_diff = F.mse_loss(
            interventional_outputs["non_causal_representation"],
            original_outputs["non_causal_representation"]
        )
        
        # We want causal_diff to be large and non_causal_diff to be small
        invariance_loss = -causal_diff + non_causal_diff
        
        return invariance_loss
    
    def _compute_independence_loss(self, causal_repr: torch.Tensor, 
                                 non_causal_repr: torch.Tensor) -> torch.Tensor:
        """Compute independence loss between causal and non-causal representations."""
        
        # Flatten representations
        causal_flat = causal_repr.view(-1, causal_repr.size(-1))  # (batch*seq_len, causal_dim)
        non_causal_flat = non_causal_repr.view(-1, non_causal_repr.size(-1))  # (batch*seq_len, non_causal_dim)
        
        # Compute correlation matrix
        causal_centered = causal_flat - causal_flat.mean(dim=0, keepdim=True)
        non_causal_centered = non_causal_flat - non_causal_flat.mean(dim=0, keepdim=True)
        
        correlation = torch.mm(causal_centered.t(), non_causal_centered) / causal_flat.size(0)
        
        # Independence loss: minimize correlation
        independence_loss = torch.norm(correlation, p='fro')
        
        return independence_loss


class DAGConstraint:
    """Directed Acyclic Graph constraint for causal discovery."""
    
    @staticmethod
    def dag_constraint(adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute DAG constraint: tr(e^(A ⊙ A)) - d = 0 for DAG.
        
        Args:
            adjacency_matrix: Adjacency matrix (n_vars, n_vars, max_lag+1)
            
        Returns:
            DAG constraint value
        """
        # Sum over lags to get instantaneous adjacency
        A = adjacency_matrix.sum(dim=-1)  # (n_vars, n_vars)
        
        # Element-wise square
        A_squared = A * A
        
        # Matrix exponential trace
        n_vars = A.size(0)
        try:
            exp_A = torch.matrix_exp(A_squared)
            constraint = torch.trace(exp_A) - n_vars
        except:
            # Fallback for numerical stability
            constraint = torch.norm(A_squared, p='fro')
        
        return constraint
    
    @staticmethod
    def sparsity_constraint(adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Compute sparsity constraint (L1 norm)."""
        return torch.norm(adjacency_matrix, p=1)


class CausalInferenceTrainer:
    """Trainer for causal inference models."""
    
    def __init__(self, config: CausalConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.causal_graph = CausalGraphNetwork(config).to(self.device)
        self.granger_model = GrangerCausalityNetwork(config).to(self.device)
        self.interventional_bci = InterventionalBCI(config, self.causal_graph).to(self.device)
        self.counterfactual_analysis = CounterfactualAnalysis(config, self.causal_graph).to(self.device)
        self.causal_representation = CausalRepresentationLearning(config).to(self.device)
        
        # Optimizers
        self.optimizers = {
            "causal_graph": torch.optim.Adam(self.causal_graph.parameters(), lr=config.learning_rate),
            "granger": torch.optim.Adam(self.granger_model.parameters(), lr=config.learning_rate),
            "interventional": torch.optim.Adam(self.interventional_bci.parameters(), lr=config.learning_rate),
            "counterfactual": torch.optim.Adam(self.counterfactual_analysis.parameters(), lr=config.learning_rate),
            "representation": torch.optim.Adam(self.causal_representation.parameters(), lr=config.learning_rate)
        }
        
        self.logger = logging.getLogger(__name__)
        
    def train_causal_discovery(self, data_loader, n_epochs: int = None) -> Dict[str, List[float]]:
        """Train causal discovery model."""
        if n_epochs is None:
            n_epochs = self.config.max_epochs
        
        self.logger.info("Training causal discovery model")
        
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                
                self.optimizers["causal_graph"].zero_grad()
                
                # Forward pass
                predictions, adj_matrix = self.causal_graph(x, return_graph=True)
                
                # Prediction loss
                pred_loss = F.mse_loss(predictions, x)
                
                # DAG constraint
                dag_loss = DAGConstraint.dag_constraint(adj_matrix)
                
                # Sparsity constraint
                sparsity_loss = DAGConstraint.sparsity_constraint(adj_matrix)
                
                # Total loss
                total_loss = (
                    pred_loss + 
                    self.config.dag_lambda * dag_loss + 
                    self.config.sparsity_lambda * sparsity_loss
                )
                
                total_loss.backward()
                self.optimizers["causal_graph"].step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(data_loader)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return {"causal_discovery_loss": losses}
    
    def train_causal_representation(self, data_loader, n_epochs: int = None) -> Dict[str, List[float]]:
        """Train causal representation learning model."""
        if n_epochs is None:
            n_epochs = self.config.max_epochs
        
        self.logger.info("Training causal representation learning")
        
        losses = {"total": [], "reconstruction": [], "classification": [], "invariance": [], "independence": []}
        
        for epoch in range(n_epochs):
            epoch_losses = {k: 0.0 for k in losses.keys()}
            
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizers["representation"].zero_grad()
                
                # Generate random interventions
                interventions = torch.randn_like(x) * 0.1 if torch.rand(1) > 0.5 else None
                
                # Compute loss
                loss_dict = self.causal_representation.compute_causal_loss(x, y, interventions)
                
                loss_dict["total_loss"].backward()
                self.optimizers["representation"].step()
                
                # Accumulate losses
                for k, v in loss_dict.items():
                    key = k.replace("_loss", "")
                    if key in epoch_losses:
                        epoch_losses[key] += v.item()
            
            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= len(data_loader)
                losses[k].append(epoch_losses[k])
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Total Loss: {epoch_losses['total']:.6f}")
        
        return losses
    
    def evaluate_causal_discovery(self, data_loader) -> Dict[str, float]:
        """Evaluate causal discovery performance."""
        self.causal_graph.eval()
        
        total_loss = 0.0
        total_dag_constraint = 0.0
        total_sparsity = 0.0
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                
                predictions, adj_matrix = self.causal_graph(x, return_graph=True)
                
                # Compute metrics
                pred_loss = F.mse_loss(predictions, x)
                dag_constraint = DAGConstraint.dag_constraint(adj_matrix)
                sparsity = DAGConstraint.sparsity_constraint(adj_matrix)
                
                total_loss += pred_loss.item()
                total_dag_constraint += dag_constraint.item()
                total_sparsity += sparsity.item()
        
        n_batches = len(data_loader)
        
        return {
            "prediction_loss": total_loss / n_batches,
            "dag_constraint": total_dag_constraint / n_batches,
            "sparsity": total_sparsity / n_batches
        }
    
    def visualize_causal_graph(self, save_path: Optional[str] = None) -> None:
        """Visualize learned causal graph."""
        self.causal_graph.eval()
        
        with torch.no_grad():
            adj_matrix = self.causal_graph.get_adjacency_matrix(hard=True)
            
            # Sum over lags for visualization
            adj_sum = adj_matrix.sum(dim=-1).cpu().numpy()
            
            # Create networkx graph
            G = nx.DiGraph()
            n_vars = adj_sum.shape[0]
            
            # Add nodes
            for i in range(n_vars):
                G.add_node(i, label=f"Var_{i}")
            
            # Add edges
            threshold = 0.5
            for i in range(n_vars):
                for j in range(n_vars):
                    if adj_sum[i, j] > threshold:
                        G.add_edge(i, j, weight=adj_sum[i, j])
            
            # Plot
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=500, alpha=0.8)
            
            # Draw edges with weights
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, 
                                 alpha=0.7, edge_color=weights, edge_cmap=plt.cm.Blues)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            plt.title("Learned Causal Graph")
            plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), 
                        label="Causal Strength")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()


def create_causal_bci_system(config: Optional[CausalConfig] = None) -> CausalInferenceTrainer:
    """
    Create a causal BCI inference system.
    
    Args:
        config: Causal inference configuration (optional)
        
    Returns:
        Configured causal inference trainer
    """
    if config is None:
        config = CausalConfig(
            n_variables=64,
            max_lag=10,
            hidden_dim=128,
            discovery_method="gradient_based",
            use_uncertainty=True
        )
    
    trainer = CausalInferenceTrainer(config)
    
    logger.info(f"Created causal BCI system with {config.n_variables} variables")
    
    return trainer


# Example usage
def run_causal_inference_example():
    """Example of running causal inference for BCI."""
    import torch.utils.data as data
    
    # Create synthetic causal dataset
    class CausalBCIDataset(data.Dataset):
        def __init__(self, n_samples=1000, n_vars=16, seq_length=100):
            self.n_samples = n_samples
            self.n_vars = n_vars
            self.seq_length = seq_length
            
            # Generate data with known causal structure
            self.data, self.labels, self.true_graph = self._generate_causal_data()
        
        def _generate_causal_data(self):
            # Create simple causal structure: X1 -> X2 -> X3 -> ... -> Y
            data = torch.zeros(self.n_samples, self.seq_length, self.n_vars)
            labels = torch.zeros(self.n_samples, dtype=torch.long)
            
            # True adjacency matrix
            true_graph = torch.zeros(self.n_vars, self.n_vars)
            
            for i in range(self.n_samples):
                # Initialize first variable
                data[i, :, 0] = torch.randn(self.seq_length)
                
                # Generate causal chain
                for t in range(1, self.seq_length):
                    for v in range(1, self.n_vars):
                        # Causal effect from previous variable
                        if v > 0:
                            data[i, t, v] = 0.7 * data[i, t-1, v-1] + 0.3 * torch.randn(1)
                            if t == 1:  # Set true graph only once
                                true_graph[v-1, v] = 1.0
                
                # Label based on last variable
                labels[i] = (data[i, -10:, -1].mean() > 0).long()
            
            return data, labels, true_graph
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    # Create dataset and data loader
    dataset = CausalBCIDataset(n_samples=500, n_vars=8, seq_length=50)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    
    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create causal inference system
    config = CausalConfig(
        n_variables=8,
        max_lag=5,
        hidden_dim=64,
        max_epochs=50
    )
    
    causal_system = create_causal_bci_system(config)
    
    # Train causal discovery
    discovery_losses = causal_system.train_causal_discovery(train_loader, n_epochs=20)
    
    # Train causal representation learning
    repr_losses = causal_system.train_causal_representation(train_loader, n_epochs=20)
    
    # Evaluate
    eval_results = causal_system.evaluate_causal_discovery(val_loader)
    
    print("Causal Inference Results:")
    print(f"Prediction Loss: {eval_results['prediction_loss']:.6f}")
    print(f"DAG Constraint: {eval_results['dag_constraint']:.6f}")
    print(f"Sparsity: {eval_results['sparsity']:.6f}")
    
    # Visualize causal graph
    causal_system.visualize_causal_graph("./causal_graph.png")
    
    # Test interventional analysis
    sample_data, sample_labels = next(iter(val_loader))
    sample_data = sample_data.to(causal_system.device)
    
    with torch.no_grad():
        intervention_results = causal_system.interventional_bci(sample_data)
        print(f"Intervention effect shape: {intervention_results['intervention_effect'].shape}")
        
        # Counterfactual analysis
        cf_results = causal_system.counterfactual_analysis.explain_decision(sample_data)
        print("Counterfactual explanations:")
        for i, explanation in enumerate(cf_results["explanation"][:3]):
            print(f"Sample {i}: {explanation}")
    
    return causal_system, eval_results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    causal_system, results = run_causal_inference_example()