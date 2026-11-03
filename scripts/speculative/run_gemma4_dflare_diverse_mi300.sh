#!/usr/bin/env bash
# Distill and train a DFlare draft on a diverse prompt corpus.
#
# Six arms established that the existing CodeAlpaca+UltraChat corpus cannot
# raise acceptance at any learning rate: the effect is monotonic in lr and
# converges to the control as lr -> 0, which is the signature of data that is
# not additive. This arm changes the prompt distribution instead of its volume,
# adding GSM8K (the current corpus contains no mathematical content at all) and
# a different code instruction set.
#
# Distillation settings are deliberately identical to the original 50K run
# (temperature 0.6, 256 new tokens, 1024 context) so the corpus is the only
# variable. Training follows the staged recipe that produced the best
# checkpoint, since bulk single-stage training on exact features was shown to be
# worse than the staged chain.
set -uo pipefail

ROOT_DIR=${ROOT_DIR:-/home/rbasu/AngelSlim}
MODEL=${MODEL:-/home/rbasu/models/gemma-4-E4B-it}
DATA_DIR=${DATA_DIR:-/home/rbasu/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/rbasu/outputs}
JOB_DIR=${JOB_DIR:-/home/rbasu/jobs/dflare-diverse}
NUM_GPUS=${NUM_GPUS:-8}

PROMPTS=$DATA_DIR/dflare_diverse50k_prompts.jsonl
DISTILLED=$DATA_DIR/dflare_diverse50k_distilled.jsonl
TRAIN_OUTPUT=$OUTPUT_ROOT/dflare-diverse50k-hf-6ep

mkdir -p "$JOB_DIR" "$OUTPUT_ROOT"
exec 9>"$JOB_DIR/pipeline.lock"
if ! flock -n 9; then
  echo "diverse pipeline already running"
  exit 0
fi

export PATH="/home/rbasu/venv-dflare/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=/home/rbasu/cache/huggingface
export TMPDIR=/home/rbasu/tmp
export HSA_OVERRIDE_GFX_VERSION=9.4.2

while pgrep -f "train_dflash_offline.py" >/dev/null; do
  echo "[wait] GPUs busy with an earlier training job; retrying in 120s"
  sleep 120
done

if [[ ! -f "$JOB_DIR/DISTILLATION_DONE" ]]; then
  echo "=== [$(date -Is)] distilling $(wc -l < "$PROMPTS") prompts ==="
  if ! torchrun --standalone --nproc-per-node="$NUM_GPUS" \
    "$ROOT_DIR/tools/generate_gemma4_self_distillation_hf_distributed.py" \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output "$DISTILLED" \
    --max-model-length 1024 \
    --max-new-tokens 256 \
    --batch-size 4 \
    --temperature 0.6 \
    --seed 0 > "$JOB_DIR/distill.log" 2>&1; then
    echo "[FAIL] distillation - see $JOB_DIR/distill.log"
    exit 1
  fi
  date -Is > "$JOB_DIR/DISTILLATION_DONE"
  echo "=== [$(date -Is)] distillation done, $(wc -l < "$DISTILLED") records ==="
fi

if [[ ! -f "$JOB_DIR/TRAINING_DONE" ]]; then
  echo "=== [$(date -Is)] HF-online training, 6 epochs ==="
  NUM_GPUS="$NUM_GPUS" \
  TARGET_MODEL_PATH="$MODEL" \
  TRAIN_DATA_PATH="$DISTILLED" \
  OUTPUT_DIR="$TRAIN_OUTPUT" \
  NUM_EPOCHS=6 \
  PER_DEVICE_BATCH_SIZE=1 \
  GRADIENT_ACCUMULATION_STEPS=8 \
  MAX_LENGTH=1024 \
  MIN_LOSS_TOKENS=16 \
  LEARNING_RATE=6e-4 \
  BLOCK_SIZE=16 \
  NUM_ANCHORS=512 \
  LOSS_DECAY_GAMMA=4.5 \
  GAMMA_WARMUP=1 \
  GAMMA_WARMUP_STEP=1 \
  DATA_NUM_PROC=8 \
  ATTENTION_BACKEND=eager \
  "$ROOT_DIR/scripts/speculative/run_gemma4_dflare_online.sh" \
    > "$JOB_DIR/train.log" 2>&1 || {
      echo "[FAIL] training - see $JOB_DIR/train.log"; exit 1; }
  date -Is > "$JOB_DIR/TRAINING_DONE"
fi

echo "=== [$(date -Is)] diverse pipeline complete: $TRAIN_OUTPUT ==="
echo "Next: generate exact vLLM features on Strix for a ~10K subset of"
echo "$DISTILLED, then run the exact-cache alignment step against it."
