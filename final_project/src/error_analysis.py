"""Export false positives and false negatives for qualitative error analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from final_project.src.config import DOCS_DIR, PROJECT_ROOT, RESULTS_DIR
from final_project.src.dataset import EOTDataset, load_splits
from final_project.src.evaluate import load_trained_model, make_eval_loader

ERROR_DIR = DOCS_DIR / "error_analysis"
FULL_RUNS = {
    "bow": RESULTS_DIR / "bow_normalized_ctx1_full",
    "bilstm": RESULTS_DIR / "bilstm_normalized_ctx1_full",
    "bert": RESULTS_DIR / "bert_normalized_ctx1_full",
}

INCOMPLETE_ENDINGS = (" and", " but", " or", " um", " uh", " so", " like")
BACKCHANNEL_TOKENS = {"yeah", "uh-huh", "um-hum", "okay", "right", "yes", "yep", "mhm", "hum"}
QUESTION_HINTS = ("what", "how", "why", "when", "where", "who", "do you", "is there", "are there")


def tag_categories(utterance: str) -> list[str]:
    tags: list[str] = []
    lower = utterance.lower().strip()
    tokens = re.findall(r"[a-zA-Z]+|'[a-z]+", lower)
    if len(tokens) <= 3:
        tags.append("short_ack")
    if tokens and all(t in BACKCHANNEL_TOKENS for t in tokens):
        tags.append("backchannel-like")
    elif any(t in BACKCHANNEL_TOKENS for t in tokens) and len(tokens) <= 5:
        tags.append("backchannel-like")
    if any(x in lower for x in (" um ", " uh ", "[noise]", " and ", " but ")):
        tags.append("trailing_disfluency")
    if any(lower.endswith(end) for end in INCOMPLETE_ENDINGS):
        tags.append("incomplete")
    if any(h in lower for h in QUESTION_HINTS) and "?" not in utterance:
        tags.append("question_without_mark")
    return tags or ["other"]


@torch.no_grad()
def collect_errors(run_dir: Path, *, max_per_type: int = 10) -> dict:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    model, vocab, tokenizer, model_type, config = load_trained_model(run_dir, str(device))
    _, _, test_examples = load_splits()
    dataset = EOTDataset(
        test_examples,
        text_mode=config["text_mode"],
        use_context=config["use_context"],
    )
    loader = make_eval_loader(model_type, dataset, vocab, tokenizer)

    fps: list[dict] = []
    fns: list[dict] = []
    idx = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        if model_type == "bow":
            logits = model(batch["features"].to(device))
        elif model_type == "bilstm":
            logits = model(batch["input_ids"].to(device))
        else:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
        preds = logits.argmax(dim=-1)

        for i in range(len(labels)):
            example = test_examples[idx]
            label = int(labels[i].item())
            pred = int(preds[i].item())
            input_text = dataset.texts[idx]
            record = {
                "context": example.context,
                "utterance": example.utterance,
                "label": label,
                "pred": pred,
                "input_text": input_text,
                "tags": tag_categories(example.utterance),
            }
            if pred == 1 and label == 0 and len(fps) < max_per_type:
                record["error_type"] = "FP"
                fps.append(record)
            elif pred == 0 and label == 1 and len(fns) < max_per_type:
                record["error_type"] = "FN"
                fns.append(record)
            idx += 1
            if len(fps) >= max_per_type and len(fns) >= max_per_type:
                return {"fp": fps, "fn": fns}

    return {"fp": fps, "fn": fns}


def render_markdown(model: str, errors: dict) -> str:
    lines = [f"# Error Analysis — {model.upper()}", "", f"Source: `{FULL_RUNS[model].name}`", ""]
    for section, key in [("False Positives (pred=EOT, label≠EOT)", "fp"), ("False Negatives (pred≠EOT, label=EOT)", "fn")]:
        lines.extend([f"## {section}", ""])
        for i, ex in enumerate(errors[key], start=1):
            tags = ", ".join(ex["tags"])
            lines.extend(
                [
                    f"### {key.upper()} {i} — `{tags}`",
                    "",
                    f"- **Label:** {ex['label']} | **Pred:** {ex['pred']}",
                    f"- **Utterance:** {ex['utterance']}",
                    f"- **Context:** {ex['context'][:200]}{'…' if len(ex['context']) > 200 else ''}",
                    f"- **Input text:** {ex['input_text'][:300]}{'…' if len(ex['input_text']) > 300 else ''}",
                    "",
                ]
            )
    return "\n".join(lines)


def export_model(model: str, run_dir: Path, out_dir: Path) -> None:
    errors = collect_errors(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{model}_errors.json"
    md_path = out_dir / f"{model}_errors.md"
    json_path.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(model, errors), encoding="utf-8")
    print(f"Wrote {md_path} ({len(errors['fp'])} FP, {len(errors['fn'])} FN)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(FULL_RUNS.keys()), nargs="*", default=list(FULL_RUNS.keys()))
    parser.add_argument("--out-dir", type=Path, default=ERROR_DIR)
    args = parser.parse_args()
    for model in args.model:
        export_model(model, FULL_RUNS[model], args.out_dir)


if __name__ == "__main__":
    main()
