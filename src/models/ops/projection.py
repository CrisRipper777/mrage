# Extracted computation from src/models/mvcge_mag_v21.py at 8f1f9c4.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



class _IntrinsicProjector(nn.Module):
    """Private modality projector for the protected intrinsic path."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(
            self.skip(x) + self.linear2(self.dropout(F.gelu(self.linear1(x))))
        )
