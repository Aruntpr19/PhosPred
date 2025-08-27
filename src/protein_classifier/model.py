"""
Neural network models for protein binding site classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PredictionModel(nn.Module):
    """
    Neural network model for predicting protein binding site types.
    
    Architecture:
    - Input: ESM2 embeddings (averaged over binding site residues)
    - Multiple fully connected layers with ReLU activations
    - Dropout for regularization
    - Output: Logits for classification
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int = 5,
                 hidden_dims: Optional[list] = None,
                 dropout_rate: float = 0.3):
        """
        Initialize the prediction model.
        
        Args:
            input_dim: Dimension of input embeddings
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(PredictionModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.hidden_dims = hidden_dims
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout (less for later layers)
            dropout = dropout_rate * (0.8 ** i) if i > 0 else dropout_rate
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Approximate size in MB
        }


class AttentionPredictionModel(nn.Module):
    """
    Advanced prediction model with attention mechanism for binding site classification.
    
    This model applies self-attention to binding site residue embeddings before
    averaging, potentially capturing important interactions between residues.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int = 5,
                 num_heads: int = 4,
                 hidden_dims: Optional[list] = None,
                 dropout_rate: float = 0.3):
        """
        Initialize the attention-based prediction model.
        
        Args:
            input_dim: Dimension of input embeddings
            num_classes: Number of output classes
            num_heads: Number of attention heads
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(AttentionPredictionModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # Build classification head
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * (0.8 ** i))
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            mask: Optional attention mask
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        if x.dim() == 2:
            # If input is already averaged, just pass through classifier
            return self.classifier(x)
        
        # Apply self-attention
        attended, _ = self.attention(x, x, x, key_padding_mask=mask)
        
        # Residual connection and layer normalization
        attended = self.layer_norm(attended + x)
        
        # Global average pooling
        if mask is not None:
            # Mask out padded positions
            mask_expanded = mask.unsqueeze(-1).expand_as(attended)
            attended = attended.masked_fill(mask_expanded, 0)
            seq_lengths = (~mask).sum(dim=1, keepdim=True).float()
            pooled = attended.sum(dim=1) / seq_lengths
        else:
            pooled = attended.mean(dim=1)
        
        return self.classifier(pooled)