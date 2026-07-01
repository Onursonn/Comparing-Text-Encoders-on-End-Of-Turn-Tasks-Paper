"""Single-example inference demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from final_project.src.config import BERT_MODEL, RESULTS_DIR, get_device
from final_project.src.dataset import load_vocab
from final_project.src.models.bert_classifier import BertEOTClassifier
from final_project.src.models.bilstm_classifier import BiLSTMClassifier
from final_project.src.models.bow_ffn import BowFFNClassifier
from final_project.src.preprocess import EOTExample, pad_sequence, prepare_text


def load_checkpoint(run_dir: Path):
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    return checkpoint


def predict_example(
    *,
    run_dir: Path,
    context: str,
    utterance: str,
) -> tuple[int, list[float]]:
    device = get_device()
    checkpoint = load_checkpoint(run_dir)
    config = checkpoint["config"]
    example = EOTExample(context=context, utterance=utterance, label=0)
    text = prepare_text(
        example,
        text_mode=config["text_mode"],
        use_context=config["use_context"],
    )

    model_type = checkpoint["model_type"]
    if model_type == "bow":
        vocab = load_vocab(Path(checkpoint["vocab_path"]))
        model = BowFFNClassifier(len(vocab))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        features = torch.tensor([vocab.binary_bow(text)], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(features)
    elif model_type == "bilstm":
        vocab = load_vocab(Path(checkpoint["vocab_path"]))
        model = BiLSTMClassifier(len(vocab))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        pad_id = vocab.token_to_id["<PAD>"]
        ids = torch.tensor(
            [pad_sequence(vocab.encode(text), 128, pad_id)],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            logits = model(ids)
    elif model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, local_files_only=True)
        model = BertEOTClassifier(BERT_MODEL)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
    else:
        raise ValueError(model_type)

    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    pred = int(logits.argmax(dim=-1).item())
    return pred, probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--utterance", required=True)
    args = parser.parse_args()
    pred, probs = predict_example(
        run_dir=args.run_dir,
        context=args.context,
        utterance=args.utterance,
    )
    print(json.dumps({"prediction": pred, "probabilities": probs}, indent=2))


if __name__ == "__main__":
    main()
