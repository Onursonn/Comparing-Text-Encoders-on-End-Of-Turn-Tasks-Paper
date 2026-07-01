"""PyTorch datasets for BoW, BiLSTM, and DistilBERT encoders."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from final_project.src.config import (
    BATCH_SIZE_BERT,
    BATCH_SIZE_BOW,
    BATCH_SIZE_LSTM,
    BERT_MODEL,
    MAX_LEN,
    MIN_FREQ,
    ftad_split_path,
)
from final_project.src.preprocess import (
    EOTExample,
    Vocabulary,
    load_ftad_split,
    pad_sequence,
    prepare_text,
)


class EOTDataset(Dataset):
    def __init__(
        self,
        examples: list[EOTExample],
        *,
        text_mode: str,
        use_context: bool,
    ) -> None:
        self.examples = examples
        self.text_mode = text_mode
        self.use_context = use_context
        self.texts = [
            prepare_text(example, text_mode=text_mode, use_context=use_context)
            for example in examples
        ]
        self.labels = [example.label for example in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def collate_bow(batch, vocab: Vocabulary):
    texts, labels = zip(*batch)
    vectors = torch.tensor([vocab.binary_bow(text) for text in texts], dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return {"features": vectors, "labels": label_tensor}


def collate_ids(batch, vocab: Vocabulary, max_len: int = MAX_LEN):
    texts, labels = zip(*batch)
    pad_id = vocab.token_to_id[Vocabulary.PAD]
    ids = torch.tensor(
        [pad_sequence(vocab.encode(text), max_len, pad_id) for text in texts],
        dtype=torch.long,
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return {"input_ids": ids, "labels": label_tensor}


def collate_bert(batch, tokenizer, max_len: int = MAX_LEN):
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoded


def build_vocab_from_train(
    train_examples: list[EOTExample],
    *,
    text_mode: str,
    use_context: bool,
) -> Vocabulary:
    texts = [
        prepare_text(example, text_mode=text_mode, use_context=use_context)
        for example in train_examples
    ]
    return Vocabulary.build(texts, min_freq=MIN_FREQ)


def save_vocab(vocab: Vocabulary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab.token_to_id, indent=2) + "\n", encoding="utf-8")


def load_vocab(path: Path) -> Vocabulary:
    token_to_id = json.loads(path.read_text(encoding="utf-8"))
    return Vocabulary(token_to_id)


def make_dataloaders(
    *,
    model_type: str,
    text_mode: str,
    use_context: bool,
    train_examples: list[EOTExample],
    val_examples: list[EOTExample],
    vocab: Vocabulary | None = None,
    tokenizer=None,
):
    train_ds = EOTDataset(train_examples, text_mode=text_mode, use_context=use_context)
    val_ds = EOTDataset(val_examples, text_mode=text_mode, use_context=use_context)

    if model_type == "bow":
        assert vocab is not None
        batch_size = BATCH_SIZE_BOW
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_bow(batch, vocab),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_bow(batch, vocab),
        )
    elif model_type == "bilstm":
        assert vocab is not None
        batch_size = BATCH_SIZE_LSTM
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_ids(batch, vocab),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_ids(batch, vocab),
        )
    elif model_type == "bert":
        assert tokenizer is not None
        batch_size = BATCH_SIZE_BERT
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_bert(batch, tokenizer),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_bert(batch, tokenizer),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return train_loader, val_loader


def load_splits() -> tuple[list[EOTExample], list[EOTExample], list[EOTExample]]:
    train = load_ftad_split(ftad_split_path("train"))
    val = load_ftad_split(ftad_split_path("valid"))
    test = load_ftad_split(ftad_split_path("test"))
    return train, val, test
