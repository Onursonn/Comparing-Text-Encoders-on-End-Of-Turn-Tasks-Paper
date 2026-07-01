"""Bag-of-words + feedforward network baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from final_project.src.config import FFN_HIDDEN


class BowFFNClassifier(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = FFN_HIDDEN, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
