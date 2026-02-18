"""Task 3: Small Transformer encoder for tabular regression (with share 15 54 3mo mean).

This module implements a Transformer-based architecture for TV viewership prediction
that treats tabular features as tokens. Unlike Task 2, this model CAN use the
3-month rolling mean feature, making it potentially more accurate.

Architecture
------------
- Each tabular feature becomes a token (1D → d_model dimensional embedding)
- Positional encoding added to capture feature order
- 1-2 Transformer encoder layers with multi-head attention
- Mean pooling over tokens + MLP regression head

Model
-----
TabularTransformer : Main model class
    - Configurable depth (num_layers), width (d_model), attention heads (nhead)
    - Default: 2 layers, 64-dim, 2 heads
    - Suitable for small to medium tabular datasets

Usage
-----
Training (see train_task3.py)::

    from src.models_task3 import TabularTransformer
    model = TabularTransformer(
        input_size=10,  # including share_15_54_3mo_mean
        d_model=64,
        nhead=2,
        num_layers=2
    )
    
Forward pass::

    >>> model = TabularTransformer(input_size=10)
    >>> x = torch.randn(32, 10)  # batch of 32 samples
    >>> output = model(x)  # shape: (32,)
    
Attention analysis::

    >>> attn_weights = model.get_attention_weights(x, layer_idx=0)
    >>> print(attn_weights.shape)  # (32, 10, 10) - attention over features

Key Features
------------
- **Explainability**: Attention weights show feature interactions
- **3mo mean usage**: Can leverage historical share information
- **Flexible architecture**: Easy to adjust depth and width
- **Positional encoding**: Sinusoidal encoding preserves feature positions

Architecture Details
--------------------
Input (B, F) → Linear projection (B, F, D)
           ↓
    + Positional encoding
           ↓
    Transformer layers (2x)
           ↓
    Mean pooling (B, D)
           ↓
    MLP head (B,) → prediction

Where B=batch, F=features, D=d_model

Advantages over MLP (Task 2)
-----------------------------
- Captures feature interactions via attention
- Uses 3-month mean feature (strong predictor)
- Attention weights provide interpretability
- Better for datasets with feature dependencies

Disadvantages
-------------
- More hyperparameters to tune
- Slower training than MLP
- May overfit on very small datasets
- Requires more careful initialization

Parameters Comparison
---------------------
Default configuration (~69K parameters):
- Input projection: F × d_model
- Transformer: 2 layers × (attention + FFN)
- Head: d_model → d_model/2 → 1

Notes
-----
- Designed for Task 3 which allows 3mo mean feature
- Attention weights saved during training for analysis
- Positional encoding uses sinusoidal functions

See Also
--------
train_task3.py : Training script for this model
models_task2.py : Alternative MLP-based models (Task 2)
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class TabularTransformer(nn.Module):
    """Transformer encoder for tabular regression with feature-level attention.
    
    Treats each feature as a token, applies multi-head self-attention to capture
    feature interactions, then pools and regresses to a single output value.
    
    Parameters
    ----------
    input_size : int
        Number of input features (e.g., 10 for Task 3 with 3mo mean).
    d_model : int, default=64
        Dimension of token embeddings and hidden states.
    nhead : int, default=2
        Number of attention heads in each layer.
    num_layers : int, default=2
        Number of Transformer encoder layers.
    dim_feedforward : int, default=128
        Dimension of feedforward network in Transformer layers.
    dropout : float, default=0.1
        Dropout probability throughout the network.
    
    Attributes
    ----------
    input_proj : nn.Linear
        Projects each scalar feature to d_model dimensions.
    transformer : nn.TransformerEncoder
        Stack of encoder layers with self-attention.
    head : nn.Sequential
        Final regression head (2 layers with ReLU).
    
    Examples
    --------
    >>> model = TabularTransformer(input_size=10, d_model=64, nhead=2)
    >>> x = torch.randn(32, 10)  # 32 samples, 10 features
    >>> predictions = model(x)  # shape: (32,)
    >>> print(predictions.shape)
    torch.Size([32])
    
    >>> # Analyze attention patterns
    >>> attn = model.get_attention_weights(x[:1], layer_idx=0)
    >>> print(attn.shape)  # (1, 10, 10) - feature-to-feature attention
    torch.Size([1, 10, 10])
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features → tokens → attention → pooling → prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, input_size).
        
        Returns
        -------
        torch.Tensor
            Predicted values, shape (batch_size,).
        """
        tokens = self.input_proj(x.unsqueeze(-1))
        tokens = tokens + self._pos_encoding(tokens.size(1), tokens.device)
        out = self.transformer(tokens)
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)

    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Extract attention weights from a specific encoder layer.
        
        Useful for interpretability: shows which features the model attends to
        when making predictions.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, input_size).
        layer_idx : int, default=0
            Which encoder layer to extract attention from (0-indexed).
        
        Returns
        -------
        torch.Tensor
            Attention weights, shape (batch_size, input_size, input_size).
            Entry [b, i, j] = attention from feature i to feature j for sample b.
            Averaged across attention heads.
        
        Examples
        --------
        >>> model = TabularTransformer(input_size=10)
        >>> x = torch.randn(1, 10)
        >>> attn = model.get_attention_weights(x)
        >>> print(f"Feature 0 attends most to feature {attn[0, 0].argmax()}")
        """
        tokens = self.input_proj(x.unsqueeze(-1))
        tokens = tokens + self._pos_encoding(tokens.size(1), tokens.device)
        layer = self.transformer.layers[layer_idx]
        _, attn_weights = layer.self_attn(
            tokens, tokens, tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        return attn_weights

    def _pos_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(1, seq_len, self.d_model, device=device)
        for i in range(seq_len):
            for j in range(0, self.d_model, 2):
                pe[0, i, j] = math.sin(i / 10000 ** (j / self.d_model))
                if j + 1 < self.d_model:
                    pe[0, i, j + 1] = math.cos(i / 10000 ** (j / self.d_model))
        return pe
