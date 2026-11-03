#!/usr/bin/env bash
# Sweep loss_decay_gamma at block_size=16.
#
# Mean committed length is 1 + sum_k prod_{j<=k} c_j, so the first conditional
# multiplies every downstream term: d(mcl)/d(c_1) is about 3.2 at the current
# operating point. The block_size=4 checkpoint reaches c_1=0.857 / c_2=0.784
# against block_size=16's 0.764 / 0.671, and the difference tracks gradient
# share: block 4 at gamma 4.5 puts 41% of the loss weight on position 1, while
# block 16 at the same gamma puts only 21% there. Lowering gamma concentrates
# weight early without giving up block depth. Positions 9-15 contribute only
# 0.167 tokens today, so trading tail accuracy for early accuracy is favourable
# even in the worst case.
#
# Warm-starts from the predecessor of the current best checkpoint so the 10K
# cache is still seen exactly once; re-running an epoch over cache the incoming
# checkpoint already consumed has previously cost held-out acceptance.
set -uo pipefail

ROOT_DIR=${ROOT_DIR:-/home/rbasu/AngelSlim}
TARGET_MODEL=${TARGET_MODEL:-/home/rbasu/models/gemma-4-E4B-it}
CACHE_DIR=${CACHE_DIR:-/home/rbasu/data/offline-cache-public10k-vllm}
INITIAL_DRAFT=${INITIAL_DRAFT:-/home/rbasu/outputs/dflare-vllm-aligned-public50k-1ep}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/rbasu/outputs}
JOB_DIR=${JOB_DIR:-/home/rbasu/jobs/dflare-gamma-sweep}
NUM_GPUS=${NUM_GPUS:-8}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
TAG=${TAG:-10k}
GAMMAS=${GAMMAS:-"2.0 3.0"}

mkdir -p "$JOB_DIR"
exec 9>"$JOB_DIR/sweep.lock"
if ! flock -n 9; then
  echo "gamma sweep already running"
  exit 0
fi

export PATH="/home/rbasu/venv-dflare/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=/home/rbasu/cache/huggingface
export TMPDIR=/home/rbasu/tmp
export HSA_OVERRIDE_GFX_VERSION=9.4.2

for GAMMA in $GAMMAS; do
  OUTPUT_DIR="$OUTPUT_ROOT/dflare-gamma${GAMMA}-${TAG}-1ep"
  if [[ -f "$OUTPUT_DIR/model.safetensors" || -f "$OUTPUT_DIR/pytorch_model.bin" ]]; then
    echo "[skip] gamma=$GAMMA already trained at $OUTPUT_DIR"
    continue
  fi
  echo "=== [$(date -Is)] training gamma=$GAMMA -> $OUTPUT_DIR ==="
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
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 2 \
    --eval_strategy no \
    --bf16 \
    --report_to none > "$JOB_DIR/train_gamma${GAMMA}.log" 2>&1; then
    echo "[FAIL] gamma=$GAMMA - see $JOB_DIR/train_gamma${GAMMA}.log"
    date -Is > "$JOB_DIR/FAILED_gamma${GAMMA}"
    continue
  fi
  echo "=== [$(date -Is)] done gamma=$GAMMA ==="
done

date -Is > "$JOB_DIR/SWEEP_DONE"
echo "=== sweep complete ==="
