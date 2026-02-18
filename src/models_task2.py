"""Task 2: Deep-learning models (MLP and LSTM) for predicting share 15 54 without 3mo mean.

This module provides neural network architectures for TV viewership share prediction.
All models are implemented in PyTorch and designed for regression tasks.

Models
------
- **MLP**: Small Multi-Layer Perceptron (2-3 hidden layers) - Primary model for Task 2
- **MLPLarge**: Larger MLP (3 hidden layers, wider) - Most complex baseline for Task 4
- **LSTMModel**: LSTM-based model for sequence modeling (experimental)

Usage
-----
Models are typically instantiated during training::

    from src.models_task2 import MLP, MLPLarge
    model = MLPLarge(input_size=9, hidden_sizes=(256, 128, 64))
    
Architecture Details
--------------------
MLP:
    - Input → Linear(128) → ReLU → Dropout(0.1)
    - → Linear(64) → ReLU → Dropout(0.1) 
    - → Linear(1)
    - Parameters: ~40K-50K depending on input size
    
MLPLarge:
    - Input → Linear(256) → ReLU → Dropout(0.15)
    - → Linear(128) → ReLU → Dropout(0.15)
    - → Linear(64) → ReLU → Dropout(0.15)
    - → Linear(1)
    - Parameters: ~60K-70K depending on input size

LSTMModel:
    - Input projection → LSTM(hidden_size=64) → Linear(1)
    - Treats tabular features as sequence (simplified approach)
    - Parameters: ~30K-40K depending on configuration

Notes
-----
- All models output a single regression value (predicted share)
- Dropout is used for regularization
- Models use ReLU activation functions
- No batch normalization (simpler architecture for small datasets)

See Also
--------
train_task2.py : Training script for these models
predict_task2.py : Inference script using trained models
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small MLP: 2-3 hidden layers, ReLU. Primary model for Task 2.
    
    A compact multi-layer perceptron for tabular regression with dropout
    regularization. Suitable for small to medium-sized feature sets.
    
    Parameters
    ----------
    input_size : int
        Number of input features (e.g., 9 for Task 2 features).
    hidden_sizes : tuple of int, default=(128, 64)
        Sizes of hidden layers. Default creates 2 hidden layers.
    dropout : float, default=0.1
        Dropout probability for regularization.
    
    Attributes
    ----------
    backbone : nn.Sequential
        Feature extraction layers with ReLU and Dropout.
    head : nn.Linear
        Final regression layer outputting single prediction.
    
    Examples
    --------
    >>> model = MLP(input_size=9, hidden_sizes=(128, 64), dropout=0.1)
    >>> x = torch.randn(32, 9)  # batch of 32 samples
    >>> output = model(x)  # shape: (32,)
    """

    def __init__(self, input_size: int, hidden_sizes: tuple = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, input_size).
        
        Returns
        -------
        torch.Tensor
            Predicted values, shape (batch_size,).
        """
        return self.head(self.backbone(x)).squeeze(-1)


class MLPLarge(nn.Module):
    """Larger MLP (most complex for Task 4): one more layer, wider.
    
    An extended multi-layer perceptron with three hidden layers for improved
    capacity. Serves as the most complex baseline model for Task 4 comparisons.
    
    Parameters
    ----------
    input_size : int
        Number of input features (e.g., 9 for Task 2 features).
    hidden_sizes : tuple of int, default=(256, 128, 64)
        Sizes of hidden layers. Default creates 3 progressively narrowing layers.
    dropout : float, default=0.15
        Dropout probability for regularization (slightly higher than MLP).
    
    Attributes
    ----------
    backbone : nn.Sequential
        Feature extraction layers with ReLU and Dropout.
    head : nn.Linear
        Final regression layer outputting single prediction.
    
    Examples
    --------
    >>> model = MLPLarge(input_size=9, hidden_sizes=(256, 128, 64))
    >>> x = torch.randn(32, 9)
    >>> output = model(x)  # shape: (32,)
    
    Notes
    -----
    - Uses higher dropout (0.15) vs MLP (0.1) due to increased capacity
    - Approximately 40-50% more parameters than standard MLP
    - Better for larger datasets but may overfit on small data
    """

    def __init__(self, input_size: int, hidden_sizes: tuple = (256, 128, 64), dropout: float = 0.15):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class LSTMModel(nn.Module):
    """
    LSTM that treats each sample as a short sequence (e.g. repeated input for compatibility).
    Most complex model option: 1-layer LSTM + linear head.
    For simplicity we use the same input repeated seq_len times as the sequence (so we don't
    require building sequences from history); in production you could feed last 7 days.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, seq_len: int = 1) -> torch.Tensor:
        # x: (B, input_size). Treat as single timestep or repeat for sequence
        if seq_len > 1:
            x = x.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            x = x.unsqueeze(1)
        x = self.proj(x)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
