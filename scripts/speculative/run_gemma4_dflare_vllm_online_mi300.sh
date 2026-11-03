#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/rbasu/AngelSlim}
PYTHON=${PYTHON:-/home/rbasu/venv-vllm/bin/python}
MODEL=${MODEL:-/home/rbasu/models/gemma-4-E4B-it}
DATA=${DATA:-/home/rbasu/data/dflare_public_50k_gemma_distilled.jsonl}
INITIAL_DRAFT=${INITIAL_DRAFT:-/home/rbasu/outputs/dflare-online-public50k-6ep}
OUTPUT_DIR=${OUTPUT_DIR:-/home/rbasu/outputs/dflare-vllm-aligned-public50k-1ep}
JOB_DIR=${JOB_DIR:-/home/rbasu/jobs/dflare-vllm-aligned}
GPU=${GPU:-0}
SAMPLE_NUM=${SAMPLE_NUM:-}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-1024}
NUM_ANCHORS=${NUM_ANCHORS:-512}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}

mkdir -p "$JOB_DIR" "$OUTPUT_DIR"
exec 9>"$JOB_DIR/training.lock"
if ! flock -n 9; then
  echo "vLLM-aligned training is already running"
  exit 0
fi

export PATH="/home/rbasu/venv-vllm/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=/home/rbasu/cache/huggingface
export TMPDIR=/home/rbasu/tmp
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export HIP_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

sample_args=()
if [[ -n "$SAMPLE_NUM" ]]; then
  sample_args+=(--sample_num "$SAMPLE_NUM")
fi

"$PYTHON" "$ROOT_DIR/tools/train_dflash_online.py" \
  --target_model_name_or_path "$MODEL" \
  --target_backend vllm \
  --draft_model_config_path "$INITIAL_DRAFT/config.json" \
  --draft_model_name_or_path "$INITIAL_DRAFT" \
  --draft_arch dflare \
  --train_data_path "$DATA" \
  --output_dir "$OUTPUT_DIR" \
  --chat_template_type gemma4 \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --min_loss_tokens 16 \
  --num_proc 8 \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --warmup_ratio 0.04 \
  --max_grad_norm 1.0 \
  --block_size 16 \
  --num_anchors "$NUM_ANCHORS" \
  --loss_decay_gamma 4.5 \
  --gamma_warmup \
  --gamma_warmup_step 1 \
  --attention_backend eager \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --max_num_seqs 1 \
  --bf16 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 2 \
  --eval_strategy no \
  --report_to none \
  "${sample_args[@]}"

date -Is > "$JOB_DIR/TRAINING_DONE"
