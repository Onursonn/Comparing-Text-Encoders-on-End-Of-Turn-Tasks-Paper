"""Hyperparameters, paths, and reproducibility defaults."""

from pathlib import Path

SEED = 42
MAX_LEN = 128
BATCH_SIZE_BOW = 32
BATCH_SIZE_LSTM = 32
BATCH_SIZE_BERT = 16
LR_BOW = 1e-3
LR_LSTM = 1e-3
LR_BERT = 2e-5
EPOCHS_BOW = 10
EPOCHS_LSTM = 10
EPOCHS_BERT = 5
EARLY_STOP_PATIENCE = 2
SUBSET_SIZE = 20_000
LOG_EVERY = 50
EMBED_DIM = 128
HIDDEN_DIM = 256
FFN_HIDDEN = 256
BERT_MODEL = "distilbert-base-uncased"
MIN_FREQ = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FTAD_DIR = DATA_DIR / "ftad"
TURNS_2K_DIR = DATA_DIR / "turns-2k"
DOCS_DIR = PROJECT_ROOT.parent / "docs"


def ftad_split_path(split: str) -> Path:
    name = "valid" if split == "val" else split
    return FTAD_DIR / "data" / "tasks" / name / "eot.txt"


def get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
