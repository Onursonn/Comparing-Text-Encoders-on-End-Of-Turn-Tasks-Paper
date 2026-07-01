"""Forward-pass smoke tests for all model types."""

import torch

from final_project.src.models.bert_classifier import BertEOTClassifier
from final_project.src.models.bilstm_classifier import BiLSTMClassifier
from final_project.src.models.bow_ffn import BowFFNClassifier


def test_bow_forward() -> None:
    model = BowFFNClassifier(vocab_size=100)
    x = torch.randn(4, 100)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_bilstm_forward() -> None:
    model = BiLSTMClassifier(vocab_size=100)
    x = torch.randint(1, 50, (4, 16))
    logits = model(x)
    assert logits.shape == (4, 2)


def test_bert_forward() -> None:
    model = BertEOTClassifier()
    input_ids = torch.ones(2, 8, dtype=torch.long)
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (2, 2)
