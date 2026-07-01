"""Training loop and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from final_project.src.config import (
    BERT_MODEL,
    DOCS_DIR,
    EARLY_STOP_PATIENCE,
    EPOCHS_BERT,
    EPOCHS_BOW,
    EPOCHS_LSTM,
    LOG_EVERY,
    LR_BERT,
    LR_BOW,
    LR_LSTM,
    RESULTS_DIR,
    SEED,
    SUBSET_SIZE,
    get_device,
)
from final_project.src.dataset import (
    build_vocab_from_train,
    load_splits,
    make_dataloaders,
    save_vocab,
)
from final_project.src.evaluate import evaluate_model, majority_baseline, progress_interval, save_metrics
from final_project.src.models.bert_classifier import BertEOTClassifier
from final_project.src.models.bilstm_classifier import BiLSTMClassifier
from final_project.src.models.bow_ffn import BowFFNClassifier
from final_project.src.preprocess import Vocabulary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_name(model: str, text_mode: str, use_context: bool, subset: int | None) -> str:
    ctx = "ctx1" if use_context else "ctx0"
    suffix = f"_n{subset}" if subset else "_full"
    return f"{model}_{text_mode}_{ctx}{suffix}"


def build_model(model_type: str, vocab: Vocabulary | None = None) -> nn.Module:
    if model_type == "bow":
        assert vocab is not None
        return BowFFNClassifier(len(vocab))
    if model_type == "bilstm":
        assert vocab is not None
        return BiLSTMClassifier(len(vocab), pad_id=vocab.token_to_id[Vocabulary.PAD])
    if model_type == "bert":
        return BertEOTClassifier(BERT_MODEL)
    raise ValueError(f"Unknown model: {model_type}")


def train_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: str,
    model_type: str,
    *,
    run_name: str,
    epoch: int,
    max_epochs: int,
    log_every: int | None = LOG_EVERY,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    total_steps = len(loader)
    interval = progress_interval(total_steps, log_every)

    for step, batch in enumerate(loader, start=1):
        labels = batch["labels"].to(device)
        optimizer.zero_grad()

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

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1

        if step == 1 or step == total_steps or step % interval == 0:
            avg_loss = total_loss / batches
            pct = 100 * step / total_steps
            print(
                f"[{run_name}] epoch {epoch}/{max_epochs} "
                f"step {step}/{total_steps} ({pct:.0f}%) loss={avg_loss:.4f}",
                flush=True,
            )

    return total_loss / max(batches, 1)


def default_epochs(model_type: str) -> int:
    if model_type == "bow":
        return EPOCHS_BOW
    if model_type == "bilstm":
        return EPOCHS_LSTM
    return EPOCHS_BERT


def default_lr(model_type: str) -> float:
    if model_type == "bow":
        return LR_BOW
    if model_type == "bilstm":
        return LR_LSTM
    return LR_BERT


def train_model(
    *,
    model_type: str,
    text_mode: str = "normalized",
    use_context: bool = True,
    subset: int | None = SUBSET_SIZE,
    epochs: int | None = None,
    patience: int = EARLY_STOP_PATIENCE,
    seed: int = SEED,
    log_every: int | None = LOG_EVERY,
) -> dict:
    set_seed(seed)
    device = get_device()
    max_epochs = epochs or default_epochs(model_type)
    lr = default_lr(model_type)
    name = run_name(model_type, text_mode, use_context, subset)
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_examples, val_examples, test_examples = load_splits()
    if subset and subset < len(train_examples):
        rng = random.Random(seed)
        train_examples = rng.sample(train_examples, subset)

    vocab = None
    tokenizer = None
    if model_type in {"bow", "bilstm"}:
        vocab = build_vocab_from_train(train_examples, text_mode=text_mode, use_context=use_context)
        save_vocab(vocab, out_dir / "vocab.json")
    else:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, local_files_only=True)

    train_loader, val_loader = make_dataloaders(
        model_type=model_type,
        text_mode=text_mode,
        use_context=use_context,
        train_examples=train_examples,
        val_examples=val_examples,
        vocab=vocab,
        tokenizer=tokenizer,
    )
    _, test_loader = make_dataloaders(
        model_type=model_type,
        text_mode=text_mode,
        use_context=use_context,
        train_examples=test_examples,
        val_examples=test_examples,
        vocab=vocab,
        tokenizer=tokenizer,
    )

    val_labels = [example.label for example in val_examples]
    baseline = majority_baseline(val_labels)
    save_metrics(baseline, out_dir / "majority_baseline_val.json")

    model = build_model(model_type, vocab).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_epoch = 0
    patience_left = patience
    history: list[dict] = []
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            model_type,
            run_name=name,
            epoch=epoch,
            max_epochs=max_epochs,
            log_every=log_every,
        )
        print(f"[{name}] epoch {epoch}/{max_epochs} validating...", flush=True)
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            model_type=model_type,
            label=f"[{name}] val",
            log_every=log_every,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics.to_dict(),
            }
        )
        print(
            f"[{name}] epoch {epoch}/{max_epochs} "
            f"loss={train_loss:.4f} val_macro_f1={val_metrics.macro_f1:.4f}"
        )

        if val_metrics.macro_f1 > best_val_f1:
            best_val_f1 = val_metrics.macro_f1
            best_epoch = epoch
            patience_left = patience
            torch.save(
                {
                    "model_type": model_type,
                    "state_dict": model.state_dict(),
                    "vocab_path": str(out_dir / "vocab.json") if vocab else None,
                    "config": {
                        "text_mode": text_mode,
                        "use_context": use_context,
                        "subset": subset,
                        "seed": seed,
                    },
                },
                out_dir / "best_model.pt",
            )
            save_metrics(val_metrics, out_dir / "best_val_metrics.json")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{name}] early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"[{name}] evaluating test set...", flush=True)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        model_type=model_type,
        label=f"[{name}] test",
        log_every=log_every,
    )
    save_metrics(test_metrics, out_dir / "test_metrics.json")

    elapsed = time.time() - start
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {
        "run": name,
        "model_type": model_type,
        "text_mode": text_mode,
        "use_context": use_context,
        "subset": subset,
        "device": device,
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "test_size": len(test_examples),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics.to_dict(),
        "majority_baseline_val": baseline.to_dict(),
        "params": param_count,
        "train_seconds": round(elapsed, 2),
        "history": history,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def append_training_log(summary: dict) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DOCS_DIR / "training_log.md"
    line = (
        f"- **{summary['run']}**: val F1={summary['best_val_macro_f1']:.4f}, "
        f"test F1={summary['test_metrics']['macro_f1']:.4f}, "
        f"{summary['train_seconds']}s, {summary['params']:,} params\n"
    )
    if not log_path.exists():
        log_path.write_text("# Training Log\n\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EOT classifiers on FTAD")
    parser.add_argument("--model", choices=["bow", "bilstm", "bert"], required=True)
    parser.add_argument("--text-mode", choices=["raw", "normalized"], default="normalized")
    parser.add_argument("--use-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subset", type=int, default=SUBSET_SIZE)
    parser.add_argument("--full", action="store_true", help="Use full training set")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = None if args.full else args.subset
    summary = train_model(
        model_type=args.model,
        text_mode=args.text_mode,
        use_context=args.use_context,
        subset=subset,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        log_every=args.log_every,
    )
    append_training_log(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
