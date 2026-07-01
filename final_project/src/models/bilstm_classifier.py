"""BiLSTM classifier (Practical 5 pattern)."""

from __future__ import annotations

import torch
import torch.nn as nn

from final_project.src.config import EMBED_DIM, HIDDEN_DIM


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = 2,
        num_classes: int = 2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, (h_n, _) = self.lstm(embedded)
        final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        return self.fc(final_hidden)
