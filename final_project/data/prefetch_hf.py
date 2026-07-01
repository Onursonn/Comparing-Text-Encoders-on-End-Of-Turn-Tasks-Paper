"""Download HuggingFace datasets and pretrained DistilBERT weights."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer

DATA = Path(__file__).resolve().parent
TURNS_DIR = DATA / "turns-2k"
STTB_DIR = DATA / "semantic-turn-taking-benchmark-ten"
MODEL_NAME = "distilbert-base-uncased"


def _save_dataset(name: str, path: Path, loader) -> None:
    if path.exists():
        ds = load_from_disk(str(path))
        rows = ds["train"].num_rows if hasattr(ds, "keys") and "train" in ds else len(ds)
        print(f"{name} already present at {path} ({rows:,} rows)")
        return
    ds = loader()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    rows = ds["train"].num_rows if hasattr(ds, "keys") and "train" in ds else len(ds)
    print(f"Saved {name} to {path} ({rows:,} rows)")


def main() -> None:
    _save_dataset(
        "TURNS-2K",
        TURNS_DIR,
        lambda: load_dataset("latishab/turns-2k"),
    )
    _save_dataset(
        "semantic-turn-taking-benchmark (ten)",
        STTB_DIR,
        lambda: load_dataset("anyreach-ai/semantic-turn-taking-benchmark", split="ten"),
    )

    print(f"Prefetching {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    print(f"  vocab={tok.vocab_size}, hidden={model.config.hidden_size}")

    manifest = {
        "ftad_dir": str((DATA / "ftad").resolve()),
        "turns_2k_dir": str(TURNS_DIR.resolve()),
        "sttb_ten_dir": str(STTB_DIR.resolve()),
        "distilbert_model": MODEL_NAME,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {DATA / 'manifest.json'}")


if __name__ == "__main__":
    main()
