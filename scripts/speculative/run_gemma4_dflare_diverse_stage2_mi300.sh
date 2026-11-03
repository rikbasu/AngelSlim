#!/usr/bin/env bash
# Stage 2 of the diverse-corpus arm: decontaminate, HF-online train, exact-align.
#
# Why this corpus matters: the 60-prompt eval set is 20 GSM8K + 20
# iamtarun/python_code_instructions + 20 UltraChat, while every checkpoint so far
# was trained on CodeAlpaca + UltraChat. Two thirds of the eval has therefore
# always been out of distribution -- no mathematical content at all, and a
# different code dataset. That is the most likely reason six arms of adding more
# of the old corpus changed nothing.
#
# Waits for the in-flight distillation, drops eval-contaminated records (the
# 32-gram filter silently passed all 10 eval prompts shorter than 32 words), then
# reproduces the staged recipe that produced the best checkpoint: 6-epoch
# HF-online training followed by exact-vLLM-feature alignment.
#
# The alignment step reuses the existing 10K exact cache, which was generated
# from the OLD corpus. That is a deliberate compromise: generating a diverse
# exact cache costs hours on Strix and would block benchmarking. The alignment
# stage's job is to move the draft from HF features to exact vLLM features, while
# the diverse content is learned in the HF-online stage. If this arm shows a
# gain, regenerate the exact cache from the diverse corpus and redo alignment.
set -uo pipefail

ROOT_DIR=${ROOT_DIR:-/home/rbasu/AngelSlim}
MODEL=${MODEL:-/home/rbasu/models/gemma-4-E4B-it}
DATA_DIR=${DATA_DIR:-/home/rbasu/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/rbasu/outputs}
JOB_DIR=${JOB_DIR:-/home/rbasu/jobs/dflare-diverse-stage2}
NUM_GPUS=${NUM_GPUS:-8}
CACHE_DIR=${CACHE_DIR:-$DATA_DIR/offline-cache-public10k-vllm}

RAW=$DATA_DIR/dflare_diverse50k_distilled.jsonl
CLEAN=$DATA_DIR/dflare_diverse50k_distilled_clean.jsonl
EVALSET=$DATA_DIR/public_mixed_60_eval_prompts.jsonl
HF_OUT=$OUTPUT_ROOT/dflare-diverse-hf-6ep
ALIGNED=$OUTPUT_ROOT/dflare-diverse-exact10k-1ep

mkdir -p "$JOB_DIR"
exec 9>"$JOB_DIR/stage2.lock"
if ! flock -n 9; then echo "stage 2 already running"; exit 0; fi

export PATH="/home/rbasu/venv-dflare/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=/home/rbasu/cache/huggingface
export TMPDIR=/home/rbasu/tmp
export HSA_OVERRIDE_GFX_VERSION=9.4.2

# --- wait for the distillation launched by the stage-1 driver ---
while pgrep -f "generate_gemma4_self_distillation" >/dev/null; do
  echo "[wait $(date +%H:%M:%S)] distillation still running"
  sleep 300
done
if [[ ! -f "$RAW" ]]; then
  echo "[FAIL] $RAW not produced by distillation"
  exit 1
fi
echo "=== [$(date -Is)] distillation finished: $(wc -l < "$RAW") records ==="

# --- decontaminate against the eval set ---
if [[ ! -f "$CLEAN" ]]; then
  echo "=== [$(date -Is)] decontaminating ==="
  python /home/rbasu/decontaminate_corpus.py "$RAW" "$CLEAN" \
    --eval-file "$EVALSET" || { echo "[FAIL] decontamination"; exit 1; }
fi
echo "=== clean corpus: $(wc -l < "$CLEAN") records ==="

# --- stage A: HF-online training on the diverse corpus ---
if [[ ! -f "$JOB_DIR/HF_DONE" ]]; then
  echo "=== [$(date -Is)] HF-online training, 6 epochs ==="
  NUM_GPUS="$NUM_GPUS" \
  TARGET_MODEL_PATH="$MODEL" \
  TRAIN_DATA_PATH="$CLEAN" \
  OUTPUT_DIR="$HF_OUT" \
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
    > "$JOB_DIR/train_hf.log" 2>&1 || { echo "[FAIL] HF stage - see train_hf.log"; exit 1; }
  date -Is > "$JOB_DIR/HF_DONE"
fi

# --- stage B: exact-vLLM-feature alignment, matching the best checkpoint recipe ---
if [[ ! -f "$JOB_DIR/ALIGN_DONE" ]]; then
  echo "=== [$(date -Is)] exact-cache alignment, 1 epoch lr 5e-5 ==="
  mkdir -p "$ALIGNED"
  torchrun --standalone --nproc-per-node="$NUM_GPUS" \
    "$ROOT_DIR/tools/train_dflash_offline.py" \
    --target_model_name_or_path "$MODEL" \
    --draft_model_config_path "$ROOT_DIR/configs/gemma4_dflare.json" \
    --init_draft_checkpoint "$HF_OUT" \
    --train_hidden_path "$CACHE_DIR" \
    --output_dir "$ALIGNED" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.04 \
    --max_grad_norm 1.0 \
    --attention_backend eager \
    --block_size 16 \
    --num_anchors 512 \
    --loss_decay_gamma 4.5 \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 2 \
    --eval_strategy no \
    --bf16 \
    --report_to none > "$JOB_DIR/train_align.log" 2>&1 \
      || { echo "[FAIL] alignment - see train_align.log"; exit 1; }
  date -Is > "$JOB_DIR/ALIGN_DONE"
fi

date -Is > "$JOB_DIR/STAGE2_DONE"
echo "=== [$(date -Is)] stage 2 complete ==="
echo "Pull $ALIGNED to Strix, convert without --reference-attention, build the"
echo "16K reduced head and the AWQ-INT4 MLP variant, then benchmark on AWQ K=3."
