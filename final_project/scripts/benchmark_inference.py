#!/usr/bin/env python3
"""Benchmark inference latency on FTAD test examples."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from final_project.src.config import DOCS_DIR, RESULTS_DIR, get_device
from final_project.src.dataset import EOTDataset, load_splits
from final_project.src.evaluate import load_trained_model, make_eval_loader

FULL_RUNS = {
    "bow": RESULTS_DIR / "bow_normalized_ctx1_full",
    "bilstm": RESULTS_DIR / "bilstm_normalized_ctx1_full",
    "bert": RESULTS_DIR / "bert_normalized_ctx1_full",
}
N_EXAMPLES = 1000
WARMUP_BATCHES = 5


@torch.no_grad()
def benchmark_run(run_dir: Path, n_examples: int = N_EXAMPLES) -> dict:
    device = get_device()
    model, vocab, tokenizer, model_type, config = load_trained_model(run_dir, device)
    _, _, test_examples = load_splits()
    test_examples = test_examples[:n_examples]
    dataset = EOTDataset(
        test_examples,
        text_mode=config["text_mode"],
        use_context=config["use_context"],
    )
    loader = make_eval_loader(model_type, dataset, vocab, tokenizer)

    for i, batch in enumerate(loader):
        labels = batch["labels"].to(device)
        if model_type == "bow":
            _ = model(batch["features"].to(device))
        elif model_type == "bilstm":
            _ = model(batch["input_ids"].to(device))
        else:
            _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        if i + 1 >= WARMUP_BATCHES:
            break

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    count = 0
    for batch in loader:
        labels = batch["labels"].to(device)
        if model_type == "bow":
            _ = model(batch["features"].to(device))
        elif model_type == "bilstm":
            _ = model(batch["input_ids"].to(device))
        else:
            _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        count += len(labels)

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    ms_per_example = (elapsed / count) * 1000
    return {
        "run": run_dir.name,
        "model_type": model_type,
        "device": device,
        "examples": count,
        "total_seconds": round(elapsed, 4),
        "ms_per_example": round(ms_per_example, 4),
    }


def main() -> None:
    results = [benchmark_run(run_dir) for run_dir in FULL_RUNS.values()]
    out = {
        "n_examples": N_EXAMPLES,
        "warmup_batches": WARMUP_BATCHES,
        "models": results,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "inference_benchmark.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
