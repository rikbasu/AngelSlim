#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/scratch/rbasu/AngelSlim}
PYTHON=${PYTHON:-/scratch/rbasu/rocm-scripts/venv-gemma-dflare/bin/python}
MODEL=${MODEL:-/proj/rdi/staff/rbasu/nobkup/models/gemma4/gemma-4-E4B-it}
DATA_ROOT=${DATA_ROOT:-/proj/rdi/staff/rbasu/nobkup/datasets/gemma4-dflare}
REMOTE=${REMOTE:-xcosdae518}
RUN_DIR=${RUN_DIR:-/scratch/rbasu/dflare-jobs/public10k}

cd /scratch/rbasu/rocm-scripts

PROMPTS=$DATA_ROOT/public_mixed_10k_prompts.jsonl
SELF_DISTILLED=$DATA_ROOT/public_mixed_10k_self_distilled.jsonl
CACHE=$DATA_ROOT/offline-cache-public10k-finite
REMOTE_CACHE=/scratch/rbasu/data/offline-cache-public10k-finite
REMOTE_OUTPUT=/scratch/rbasu/outputs/dflare-offline-public10k-finite-6ep
REMOTE_JOB=/scratch/rbasu/jobs/dflare-public10k

mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR/PIPELINE_FAILED"
trap 'date -Is > "$RUN_DIR/PIPELINE_FAILED"' ERR

export VLLM_TARGET_DEVICE=rocm
export PYTHONPATH="/scratch/rbasu/rocm-scripts/venv-gemma-dflare/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi:$ROOT_DIR"

"$PYTHON" - <<'PY'
import torch

assert torch.cuda.is_available(), "ROCm GPU is not visible"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print("GPU preflight:", torch.cuda.get_device_name(0))
PY

if [[ ! -f "$RUN_DIR/PROMPTS_DONE" ]]; then
  "$PYTHON" "$ROOT_DIR/tools/prepare_public_gemma4_prompts.py" \
    --output "$PROMPTS" --count 10000 --seed 42
  date -Is > "$RUN_DIR/PROMPTS_DONE"
fi

if [[ ! -f "$RUN_DIR/SELF_DISTILLATION_DONE" ]]; then
  "$PYTHON" "$ROOT_DIR/tools/generate_gemma4_self_distillation.py" \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output "$SELF_DISTILLED" \
    --max-model-len 1024 \
    --max-new-tokens 128 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --seed 0
  date -Is > "$RUN_DIR/SELF_DISTILLATION_DONE"
fi

if [[ ! -f "$RUN_DIR/CACHE_DONE" ]]; then
  "$PYTHON" "$ROOT_DIR/tools/generate_dflash_data.py" \
    --target_model_name_or_path "$MODEL" \
    --draft_model_config_path "$ROOT_DIR/configs/gemma4_dflare.json" \
    --train_data_path "$SELF_DISTILLED" \
    --output_dir "$CACHE" \
    --chat_template_type gemma4 \
    --model_max_length 1024 \
    --min_loss_tokens 16 \
    --batch_size 1 \
    --num_proc 8 \
    --shard_size 100
  date -Is > "$RUN_DIR/CACHE_DONE"
fi

ssh -o BatchMode=yes "$REMOTE" \
  "test ! -e '$REMOTE_OUTPUT'; mkdir -p '$REMOTE_CACHE' '$REMOTE_JOB'"
rsync -a --delete --exclude='.git' --exclude='outputs' \
  "$ROOT_DIR/" "$REMOTE:/scratch/rbasu/AngelSlim/"
rsync -a --delete "$CACHE/" "$REMOTE:$REMOTE_CACHE/"
date -Is > "$RUN_DIR/TRANSFER_DONE"

ssh -o BatchMode=yes "$REMOTE" \
  "tmux has-session -t dflare-public10k-mi100 2>/dev/null || \
   tmux new-session -d -s dflare-public10k-mi100 \
   'bash /scratch/rbasu/AngelSlim/scripts/speculative/run_gemma4_dflare_offline_mi100.sh \
   $REMOTE_CACHE $REMOTE_OUTPUT 6 > $REMOTE_JOB/train.log 2>&1'"

date -Is > "$RUN_DIR/REMOTE_TRAINING_STARTED"
