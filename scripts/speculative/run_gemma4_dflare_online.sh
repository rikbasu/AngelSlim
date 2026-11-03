#!/usr/bin/env bash
set -euo pipefail

# Gemma-4-E4B DFlare online training.
#
# Required:
#   TARGET_MODEL_PATH=/path/to/google/gemma-4-E4B-it
#   TRAIN_DATA_PATH=/path/to/conversations.jsonl
# Optional:
#   NUM_GPUS=1, OUTPUT_DIR=..., SAMPLE_NUM=5000
#
# The target model must be loadable by Transformers 5.14.1 or newer and the
# training data must use the standard {"id", "conversations"} schema.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

NUM_GPUS="${NUM_GPUS:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-eager}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
DRAFT_CONFIG_PATH="${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/gemma4_dflare.json}"
DRAFT_MODEL_NAME_OR_PATH="${DRAFT_MODEL_NAME_OR_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/gemma4-e4b-dflare-b16}"
SAMPLE_NUM="${SAMPLE_NUM:-}"
DATA_NUM_PROC="${DATA_NUM_PROC:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
MIN_LOSS_TOKENS="${MIN_LOSS_TOKENS:-16}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-7}"
MAX_TRAIN_POSITIONS="${MAX_TRAIN_POSITIONS:-}"
AR_AUX_WEIGHT="${AR_AUX_WEIGHT:-0}"
ACCEPTANCE_AWARE_LOSS="${ACCEPTANCE_AWARE_LOSS:-0}"
DPACE_ALPHA="${DPACE_ALPHA:-0.5}"
HORIZON_BOUNDARY="${HORIZON_BOUNDARY:-}"
HORIZON_TAIL_WEIGHT="${HORIZON_TAIL_WEIGHT:-}"
FRONT_TEACHER_DRAFT_PATH="${FRONT_TEACHER_DRAFT_PATH:-}"
FRONT_TEACHER_BLOCK_SIZE="${FRONT_TEACHER_BLOCK_SIZE:-16}"
FRONT_DISTILL_POSITIONS="${FRONT_DISTILL_POSITIONS:-15}"
FRONT_DISTILL_WEIGHT="${FRONT_DISTILL_WEIGHT:-0}"
FRONT_DISTILL_TEMPERATURE="${FRONT_DISTILL_TEMPERATURE:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_STEPS="${MAX_STEPS:-}"
TARGET_BACKEND="${TARGET_BACKEND:-hf}"
TARGET_GPU_MEMORY_UTILIZATION="${TARGET_GPU_MEMORY_UTILIZATION:-0.5}"
TARGET_MAX_NUM_SEQS="${TARGET_MAX_NUM_SEQS:-1}"
DATASET_MODAL_TYPE="${DATASET_MODAL_TYPE:-LLM}"
GAMMA_WARMUP="${GAMMA_WARMUP:-0}"
GAMMA_WARMUP_STEP="${GAMMA_WARMUP_STEP:-1}"

if [[ -z "$TARGET_MODEL_PATH" || -z "$TRAIN_DATA_PATH" ]]; then
  echo "TARGET_MODEL_PATH and TRAIN_DATA_PATH must be set" >&2
  exit 2
fi

SAMPLE_ARGS=()
if [[ -n "$SAMPLE_NUM" ]]; then
  SAMPLE_ARGS+=(--sample_num "$SAMPLE_NUM")
fi
INIT_ARGS=()
if [[ -n "$DRAFT_MODEL_NAME_OR_PATH" ]]; then
  INIT_ARGS+=(--draft_model_name_or_path "$DRAFT_MODEL_NAME_OR_PATH")
fi
POSITION_ARGS=()
if [[ -n "$MAX_TRAIN_POSITIONS" ]]; then
  POSITION_ARGS+=(--max_train_positions "$MAX_TRAIN_POSITIONS")
fi
GAMMA_ARGS=()
if [[ "$GAMMA_WARMUP" == "1" ]]; then
  GAMMA_ARGS+=(--gamma_warmup --gamma_warmup_step "$GAMMA_WARMUP_STEP")
fi
DPACE_ARGS=()
if [[ "$ACCEPTANCE_AWARE_LOSS" == "1" ]]; then
  DPACE_ARGS+=(--acceptance_aware_loss --dpace_alpha "$DPACE_ALPHA")
fi
HORIZON_ARGS=()
if [[ -n "$HORIZON_BOUNDARY" || -n "$HORIZON_TAIL_WEIGHT" ]]; then
  if [[ -z "$HORIZON_BOUNDARY" || -z "$HORIZON_TAIL_WEIGHT" ]]; then
    echo "HORIZON_BOUNDARY and HORIZON_TAIL_WEIGHT must be set together" >&2
    exit 2
  fi
  HORIZON_ARGS+=(
    --horizon_boundary "$HORIZON_BOUNDARY"
    --horizon_tail_weight "$HORIZON_TAIL_WEIGHT"
  )
fi
MAX_STEPS_ARGS=()
if [[ -n "$MAX_STEPS" ]]; then
  MAX_STEPS_ARGS+=(--max_steps "$MAX_STEPS")
fi
FRONT_DISTILL_ARGS=()
if [[ -n "$FRONT_TEACHER_DRAFT_PATH" ]]; then
  FRONT_DISTILL_ARGS+=(
    --front_teacher_draft_path "$FRONT_TEACHER_DRAFT_PATH"
    --front_teacher_block_size "$FRONT_TEACHER_BLOCK_SIZE"
    --front_distill_positions "$FRONT_DISTILL_POSITIONS"
    --front_distill_weight "$FRONT_DISTILL_WEIGHT"
    --front_distill_temperature "$FRONT_DISTILL_TEMPERATURE"
  )
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  DISTRIBUTED_ARGS=(--standalone --nproc_per_node "$NUM_GPUS")
  FSDP_ARGS=(
    --fsdp "shard_grad_op auto_wrap"
    --fsdp_config "$ROOT_DIR/configs/fsdp_config.json"
  )
else
  DISTRIBUTED_ARGS=(--nproc_per_node 1)
  FSDP_ARGS=()
fi

torchrun "${DISTRIBUTED_ARGS[@]}" \
  "$ROOT_DIR/tools/train_dflash_online.py" \
  --target_model_name_or_path "$TARGET_MODEL_PATH" \
  --target_backend "$TARGET_BACKEND" \
  --gpu_memory_utilization "$TARGET_GPU_MEMORY_UTILIZATION" \
  --max_num_seqs "$TARGET_MAX_NUM_SEQS" \
  --draft_model_config_path "$DRAFT_CONFIG_PATH" \
  "${INIT_ARGS[@]}" \
  --draft_arch dflare \
  --train_data_path "$TRAIN_DATA_PATH" \
  --dataset_modal_type "$DATASET_MODAL_TYPE" \
  --output_dir "$OUTPUT_DIR" \
  --modal_type DFlash \
  --training_mode online \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --warmup_ratio 0.04 \
  --max_grad_norm 1.0 \
  --model_max_length "$MAX_LENGTH" \
  --min_loss_tokens "$MIN_LOSS_TOKENS" \
  --chat_template_type gemma4 \
  --attention_backend "$ATTENTION_BACKEND" \
  --block_size "$BLOCK_SIZE" \
  --num_anchors "$NUM_ANCHORS" \
  --loss_decay_gamma "$LOSS_DECAY_GAMMA" \
  --ar_aux_weight "$AR_AUX_WEIGHT" \
  "${DPACE_ARGS[@]}" \
  "${HORIZON_ARGS[@]}" \
  "${FRONT_DISTILL_ARGS[@]}" \
  "${POSITION_ARGS[@]}" \
  --num_proc "$DATA_NUM_PROC" \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  "${MAX_STEPS_ARGS[@]}" \
  --bf16 \
  --lr_scheduler_type cosine \
  --dataloader_drop_last \
  "${FSDP_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}" \
  "${GAMMA_ARGS[@]}"
