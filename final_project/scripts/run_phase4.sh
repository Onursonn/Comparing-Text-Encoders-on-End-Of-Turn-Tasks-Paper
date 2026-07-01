#!/usr/bin/env bash
# Phase 4 experiment batch: context ablation, raw vs normalized, OOD eval.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
LOG="$REPO/docs/phase4_runs.log"
VENV="$ROOT/.venv/bin/activate"

echo "=== Phase 4 started $(date) ===" | tee -a "$LOG"
cd "$REPO"
source "$VENV"
export PYTORCH_ENABLE_MPS_FALLBACK=1

SUBSET=20000
COMMON="--subset $SUBSET --log-every 50"

run_train() {
  echo "" | tee -a "$LOG"
  echo ">>> TRAIN $*" | tee -a "$LOG"
  python -m final_project.src.train "$@" 2>&1 | tee -a "$LOG"
}

run_ood() {
  local dir="$1"
  echo "" | tee -a "$LOG"
  echo ">>> OOD $dir" | tee -a "$LOG"
  python -m final_project.src.evaluate --run-dir "$dir" --ood --log-every 50 2>&1 | tee -a "$LOG"
}

# --- Context ablation (normalized, no context) ---
for model in bow bilstm bert; do
  run_train --model "$model" --text-mode normalized --no-use-context $COMMON
done

# --- Robustness: raw text (context on) ---
for model in bow bilstm bert; do
  run_train --model "$model" --text-mode raw --use-context $COMMON
done

# --- OOD on all 20k main + ablation + raw runs ---
for dir in "$ROOT/results"/*_n20000; do
  if [[ -f "$dir/best_model.pt" ]]; then
    run_ood "$dir"
  fi
done

echo "=== Phase 4 finished $(date) ===" | tee -a "$LOG"
