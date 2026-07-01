"""Evaluation metrics, baselines, and OOD eval."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from final_project.src.config import (
    BATCH_SIZE_BERT,
    BATCH_SIZE_BOW,
    BATCH_SIZE_LSTM,
    BERT_MODEL,
    LOG_EVERY,
    MAX_LEN,
    TURNS_2K_DIR,
    get_device,
)
from final_project.src.dataset import collate_bert, collate_bow, collate_ids, load_vocab
from final_project.src.models.bert_classifier import BertEOTClassifier
from final_project.src.models.bilstm_classifier import BiLSTMClassifier
from final_project.src.models.bow_ffn import BowFFNClassifier
from final_project.src.preprocess import EOTExample, prepare_text


@dataclass
class EvaluationResult:
    accuracy: float
    macro_f1: float
    precision: float
    recall: float
    confusion: list[list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


def majority_baseline(labels: list[int]) -> EvaluationResult:
    majority = max(set(labels), key=labels.count)
    preds = [majority] * len(labels)
    return compute_metrics(labels, preds)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> EvaluationResult:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        precision=float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        confusion=cm,
    )


def save_metrics(result: EvaluationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")


def progress_interval(total: int, log_every: int | None) -> int:
    if log_every is not None:
        return max(1, log_every)
    return max(1, total // 20)


def log_progress(label: str, step: int, total: int, extra: str = "") -> None:
    pct = 100 * step / total
    suffix = f" {extra}" if extra else ""
    print(f"{label} step {step}/{total} ({pct:.0f}%){suffix}", flush=True)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: str,
    *,
    model_type: str,
    label: str = "eval",
    log_every: int | None = LOG_EVERY,
) -> EvaluationResult:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total = len(loader)
    interval = progress_interval(total, log_every)

    for step, batch in enumerate(loader, start=1):
        labels = batch["labels"].to(device)
        if model_type == "bow":
            features = batch["features"].to(device)
            logits = model(features)
        elif model_type == "bilstm":
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
        elif model_type == "bert":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        preds = logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        if step == 1 or step == total or step % interval == 0:
            log_progress(label, step, total)

    return compute_metrics(y_true, y_pred)


class Turns2KDataset(Dataset):
    def __init__(self, *, text_mode: str, use_context: bool) -> None:
        rows = load_from_disk(str(TURNS_2K_DIR))["train"]
        self.examples = [
            EOTExample(context="", utterance=row["content"], label=int(row["label"]))
            for row in rows
        ]
        self.text_mode = text_mode
        self.use_context = use_context
        self.texts = [
            prepare_text(example, text_mode=text_mode, use_context=use_context)
            for example in self.examples
        ]
        self.labels = [example.label for example in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def load_trained_model(run_dir: Path, device: str):
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model_type = checkpoint["model_type"]
    config = checkpoint["config"]
    vocab = None
    tokenizer = None

    if model_type == "bow":
        vocab = load_vocab(Path(checkpoint["vocab_path"]))
        model = BowFFNClassifier(len(vocab)).to(device)
    elif model_type == "bilstm":
        vocab = load_vocab(Path(checkpoint["vocab_path"]))
        model = BiLSTMClassifier(len(vocab)).to(device)
    elif model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, local_files_only=True)
        model = BertEOTClassifier(BERT_MODEL).to(device)
    else:
        raise ValueError(model_type)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocab, tokenizer, model_type, config


def make_eval_loader(model_type: str, dataset: Dataset, vocab, tokenizer):
    if model_type == "bow":
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE_BOW,
            shuffle=False,
            collate_fn=lambda batch: collate_bow(batch, vocab),
        )
    if model_type == "bilstm":
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE_LSTM,
            shuffle=False,
            collate_fn=lambda batch: collate_ids(batch, vocab, MAX_LEN),
        )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE_BERT,
        shuffle=False,
        collate_fn=lambda batch: collate_bert(batch, tokenizer, MAX_LEN),
    )


def evaluate_ood(run_dir: Path, *, log_every: int | None = LOG_EVERY) -> EvaluationResult:
    device = get_device()
    model, vocab, tokenizer, model_type, config = load_trained_model(run_dir, device)
    dataset = Turns2KDataset(
        text_mode=config["text_mode"],
        use_context=config["use_context"],
    )
    loader = make_eval_loader(model_type, dataset, vocab, tokenizer)
    print(f"[ood] evaluating {run_dir.name} on TURNS-2K ({len(dataset)} examples)", flush=True)
    metrics = evaluate_model(
        model,
        loader,
        device,
        model_type=model_type,
        label=f"[ood:{run_dir.name}]",
        log_every=log_every,
    )
    save_metrics(metrics, run_dir / "ood_turns2k_metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained EOT models")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ood", action="store_true", help="Zero-shot eval on TURNS-2K")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ood:
        metrics = evaluate_ood(args.run_dir, log_every=args.log_every)
        print(json.dumps(metrics.to_dict(), indent=2))
        return
    raise SystemExit("Specify --ood (in-domain eval is handled during training)")


if __name__ == "__main__":
    main()
