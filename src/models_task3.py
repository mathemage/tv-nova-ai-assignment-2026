"""
Task 3: Small Transformer encoder for tabular regression (with share 15 54 3mo mean).
Each row is a sequence of feature embeddings; 1-2 encoder layers; mean pool + regression head.
Explainable via attention weights.
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class TabularTransformer(nn.Module):
    """
    Transformer encoder on tabular features: each feature is a token (linear projection),
    then 1-2 TransformerEncoderLayer, mean pool, linear head for regression.
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
        """x: (B, input_size). Each feature becomes a token."""
        tokens = self.input_proj(x.unsqueeze(-1))
        tokens = tokens + self._pos_encoding(tokens.size(1), tokens.device)
        out = self.transformer(tokens)
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)

    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Return attention weights from the first encoder layer (B, nhead, seq, seq)."""
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
