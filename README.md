# End-of-Turn Detection — Final Project

**Course:** Python for NLP (HHU)  
**Question:** How well can simple text models tell when someone has **finished speaking**?

---

## What is this?

Imagine a conversation: you wait until the other person **stops talking** — only then do you reply.

That is what this project teaches a computer to do: from the **written text** of a sentence (plus a bit of context from earlier turns), guess whether the speaker is **still going** or **done**.

We compare **three models** — from very simple to a bit smarter:

| Model | In plain terms |
|-------|----------------|
| **BoW + FFN** | Counts words and decides from that |
| **BiLSTM** | Reads the sentence word by word and remembers word order |
| **DistilBERT** | Pre-trained language model, fine-tuned for this task |

All models train on **FTAD** (dialogue data). We test on FTAD and also on **TURNS-2K** (different data, to see whether the model truly generalises or just memorised).

---

## Quick start

Everything runs from the `final_project/` folder.

### 1. Set up

```bash
cd final_project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash data/download.sh                # downloads FTAD + TURNS-2K
export PYTORCH_ENABLE_MPS_FALLBACK=1 # macOS Apple Silicon only
```

### 2. Check that it works

From the **project root** (`python-thesis/`):

```bash
pytest final_project/test/ -q
```

If all tests pass → setup is good.

### 3. Train a model

From the **project root**, with the venv active:

```bash
# Quick (20,000 examples — minutes instead of hours):
python -m final_project.src.train --model bilstm --subset 20000 --text-mode normalized --use-context

# Full (entire FTAD — takes longer):
python -m final_project.src.train --model bert --full --text-mode normalized --use-context
```

Valid `--model` values: `bow`, `bilstm`, `bert`.

Outputs go to `final_project/results/<run_name>/`.

---

## Useful commands

| Task | Command |
|------|---------|
| Train all three main models (full) | `bash final_project/scripts/run_full_training.sh` |
| Quick experiments | `bash final_project/scripts/run_phase4.sh` |
| Out-of-domain evaluation | `python -m final_project.src.evaluate --run-dir final_project/results/bert_normalized_ctx1_full --ood` |
| Plot figures | `python final_project/scripts/plot_results.py` |

Run all commands from the project root with the venv active.

---

## Results (main run, FTAD test)

| Model | macro-F1 |
|-------|--------:|
| BoW + FFN | 0.638 |
| BiLSTM | 0.677 |
| DistilBERT | 0.700 |

DistilBERT wins — but BiLSTM is already close, with less compute.

---

## Project layout

```
python-thesis/
├── README.md
└── final_project/
    ├── src/                  ← training & evaluation code
    ├── scripts/              ← batch scripts & plots
    ├── test/                 ← pytest tests
    ├── data/                 ← download scripts (datasets not in repo)
    ├── figures/              ← generated plots (local)
    └── results/              ← training outputs (local, not in repo)
```

Settings (seed, paths, hyperparameters): `final_project/src/config.py`
