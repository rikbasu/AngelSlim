#!/usr/bin/env bash
# Train with anchors drawn from the distribution serving actually visits.
#
# Instrumenting the runtime shows that only a request's first block starts at a
# freely chosen anchor: that block commits 6.84 tokens, while every later block
# starts where the previous one was rejected and commits 3.27. Rejection anchors
# are 1112 of 1173 steps on the 60-prompt set, so uniform-anchor training
# optimizes a distribution that is 5% of serving. This relocates a configurable
# fraction of anchors to each block's first mismatch.
#
# Settings below match the current best checkpoint
# (dflare-strix023-cache-aligned-10k-1ep: 10K cache, warm start from
# dflare-vllm-aligned-public50k-1ep, lr 5e-5, gamma 4.5, 1 epoch) so that
# hard_anchor_fraction is the only variable.
set -uo pipefail

ROOT_DIR=${ROOT_DIR:-/home/rbasu/AngelSlim}
TARGET_MODEL=${TARGET_MODEL:-/home/rbasu/models/gemma-4-E4B-it}
CACHE_DIR=${CACHE_DIR:-/home/rbasu/data/offline-cache-public10k-vllm}
INITIAL_DRAFT=${INITIAL_DRAFT:-/home/rbasu/outputs/dflare-vllm-aligned-public50k-1ep}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/rbasu/outputs}
JOB_DIR=${JOB_DIR:-/home/rbasu/jobs/dflare-hard-anchor}
NUM_GPUS=${NUM_GPUS:-8}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
GAMMA=${GAMMA:-4.5}
FRACTIONS=${FRACTIONS:-"1.0 0.5"}

mkdir -p "$JOB_DIR"
exec 9>"$JOB_DIR/sweep.lock"
if ! flock -n 9; then
  echo "hard-anchor sweep already running"
  exit 0
fi

export PATH="/home/rbasu/venv-dflare/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=/home/rbasu/cache/huggingface
export TMPDIR=/home/rbasu/tmp
export HSA_OVERRIDE_GFX_VERSION=9.4.2

# Do not contend with an in-flight training job for the GPUs.
while pgrep -f "train_dflash_offline.py" >/dev/null; do
  echo "[wait] GPUs busy with an earlier training job; retrying in 120s"
  sleep 120
done

for FRAC in $FRACTIONS; do
  OUTPUT_DIR="$OUTPUT_ROOT/dflare-hardanchor${FRAC}-10k-1ep"
  if [[ -f "$OUTPUT_DIR/model.safetensors" ]]; then
    echo "[skip] fraction=$FRAC already trained"
    continue
  fi
  echo "=== [$(date -Is)] training hard_anchor_fraction=$FRAC -> $OUTPUT_DIR ==="
  mkdir -p "$OUTPUT_DIR"
  if ! torchrun --standalone --nproc-per-node="$NUM_GPUS" \
    "$ROOT_DIR/tools/train_dflash_offline.py" \
    --target_model_name_or_path "$TARGET_MODEL" \
    --draft_model_config_path "$ROOT_DIR/configs/gemma4_dflare.json" \
    --init_draft_checkpoint "$INITIAL_DRAFT" \
    --train_hidden_path "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio 0.04 \
    --max_grad_norm 1.0 \
    --attention_backend eager \
    --block_size 16 \
    --num_anchors 512 \
    --loss_decay_gamma "$GAMMA" \
    --hard_anchor_fraction "$FRAC" \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 2 \
    --eval_strategy no \
    --bf16 \
    --report_to none > "$JOB_DIR/train_frac${FRAC}.log" 2>&1; then
    echo "[FAIL] fraction=$FRAC - see $JOB_DIR/train_frac${FRAC}.log"
    continue
  fi
  echo "=== [$(date -Is)] done fraction=$FRAC ==="
done

date -Is > "$JOB_DIR/SWEEP_DONE"
echo "=== hard-anchor sweep complete ==="
