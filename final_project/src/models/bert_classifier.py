"""DistilBERT fine-tune classifier (Practical 8 pattern)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from final_project.src.config import BERT_MODEL


class BertEOTClassifier(nn.Module):
    def __init__(self, model_name: str = BERT_MODEL, num_classes: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_state = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_state)
