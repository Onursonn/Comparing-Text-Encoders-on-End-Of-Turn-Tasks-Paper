#!/usr/bin/env bash
# Full FTAD training: main comparison (normalized, context on) + OOD eval.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
LOG="$REPO/docs/full_training.log"
VENV="$ROOT/.venv/bin/activate"
COMMON="--full --text-mode normalized --use-context --log-every 100"

echo "=== Full training started $(date) ===" | tee "$LOG"
cd "$REPO"
source "$VENV"
export PYTORCH_ENABLE_MPS_FALLBACK=1

run_train() {
  local model="$1"
  local out="$ROOT/results/${model}_normalized_ctx1_full"
  if [[ -f "$out/best_model.pt" && -f "$out/test_metrics.json" ]]; then
    echo ">>> SKIP $model (already done)" | tee -a "$LOG"
    return 0
  fi
  echo "" | tee -a "$LOG"
  echo ">>> TRAIN $model $COMMON" | tee -a "$LOG"
  python -m final_project.src.train --model "$model" $COMMON 2>&1 | tee -a "$LOG"
}

run_ood() {
  local dir="$1"
  if [[ -f "$dir/ood_turns2k_metrics.json" ]]; then
    echo ">>> SKIP OOD $dir" | tee -a "$LOG"
    return 0
  fi
  echo "" | tee -a "$LOG"
  echo ">>> OOD $dir" | tee -a "$LOG"
  python -m final_project.src.evaluate --run-dir "$dir" --ood --log-every 50 2>&1 | tee -a "$LOG"
}

for model in bow bilstm bert; do
  run_train "$model"
done

for model in bow bilstm bert; do
  run_ood "$ROOT/results/${model}_normalized_ctx1_full"
done

echo "=== Full training finished $(date) ===" | tee -a "$LOG"
