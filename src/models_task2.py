"""
Task 2: Deep-learning models (MLP and LSTM) for predicting share 15 54 without 3mo mean.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small MLP: 2-3 hidden layers, ReLU. Primary model for Task 2."""

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
        return self.head(self.backbone(x)).squeeze(-1)


class MLPLarge(nn.Module):
    """Larger MLP (most complex for Task 4): one more layer, wider."""

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
