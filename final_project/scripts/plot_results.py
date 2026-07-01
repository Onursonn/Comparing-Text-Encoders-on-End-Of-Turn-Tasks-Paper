#!/usr/bin/env python3
"""Generate report figures from saved metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RESULTS = REPO / "final_project" / "results"
FIGURES = REPO / "report" / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fig1_encoder_ladder() -> None:
    models = ["BoW+FFN", "BiLSTM", "DistilBERT"]
    runs = [
        "bow_normalized_ctx1_full",
        "bilstm_normalized_ctx1_full",
        "bert_normalized_ctx1_full",
    ]
    f1 = [load_json(RESULTS / r / "test_metrics.json")["macro_f1"] for r in runs]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, f1, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("macro-F1 (FTAD test)")
    ax.set_title("Encoder Ladder — Full FTAD, normalized, context on")
    ax.set_ylim(0.5, 0.75)
    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig1_encoder_ladder.pdf")
    fig.savefig(FIGURES / "fig1_encoder_ladder.png", dpi=150)
    plt.close(fig)


def fig2_raw_vs_normalized() -> None:
    models = ["BoW", "BiLSTM", "DistilBERT"]
    normalized = []
    raw = []
    for m in ("bow", "bilstm", "bert"):
        normalized.append(load_json(RESULTS / f"{m}_normalized_ctx1_n20000" / "test_metrics.json")["macro_f1"])
        raw.append(load_json(RESULTS / f"{m}_raw_ctx1_n20000" / "test_metrics.json")["macro_f1"])
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, normalized, width, label="normalized", color="#4C72B0")
    ax.bar(x + width / 2, raw, width, label="raw", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("macro-F1 (FTAD test)")
    ax.set_title("Punctuation Robustness — 20k subset, context on")
    ax.legend()
    ax.set_ylim(0.45, 0.75)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_raw_vs_normalized.pdf")
    fig.savefig(FIGURES / "fig2_raw_vs_normalized.png", dpi=150)
    plt.close(fig)


def fig3_confusion_matrices() -> None:
    runs = [
        ("BiLSTM", "bilstm_normalized_ctx1_full"),
        ("DistilBERT", "bert_normalized_ctx1_full"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, (name, run) in zip(axes, runs):
        cm = np.array(load_json(RESULTS / run / "test_metrics.json")["confusion"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["not EOT", "EOT"])
        ax.set_yticklabels(["not EOT", "EOT"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.suptitle("Confusion Matrices — Full FTAD test")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_confusion.pdf")
    fig.savefig(FIGURES / "fig3_confusion.png", dpi=150)
    plt.close(fig)


def main() -> None:
    fig1_encoder_ladder()
    fig2_raw_vs_normalized()
    fig3_confusion_matrices()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
