#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/scratch/rbasu/AngelSlim}
PYTHON=${PYTHON:-/scratch/rbasu/rocm-scripts/venv-gemma-dflare/bin/python}
MODEL=${MODEL:-/proj/rdi/staff/rbasu/nobkup/models/gemma4/gemma-4-E4B-it}
DATA_ROOT=${DATA_ROOT:-/proj/rdi/staff/rbasu/nobkup/datasets/gemma4-dflare}
RUN_DIR=${RUN_DIR:-/scratch/rbasu/dflare-jobs/public50k}
EVAL_PROMPTS=${EVAL_PROMPTS:-$DATA_ROOT/public_mixed_60_eval_prompts.jsonl}

PROMPTS=$DATA_ROOT/dflare_public_50k_prompts.jsonl
SELF_DISTILLED=$DATA_ROOT/dflare_public_50k_gemma_distilled.jsonl
CACHE=$DATA_ROOT/offline-cache-public50k-vllm

mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR/PIPELINE_FAILED"
trap 'date -Is > "$RUN_DIR/PIPELINE_FAILED"' ERR

export VLLM_TARGET_DEVICE=rocm
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONPATH="/scratch/rbasu/rocm-scripts/venv-gemma-dflare/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi:$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/scratch/rbasu/cache/huggingface}
export TMPDIR=${TMPDIR:-/scratch/rbasu/tmp}

if [[ ! -f "$RUN_DIR/PROMPTS_DONE" ]]; then
  test ! -e "$PROMPTS"
  "$PYTHON" "$ROOT_DIR/tools/prepare_public_gemma4_prompts.py" \
    --recipe dflare_open \
    --output "$PROMPTS" \
    --count 50000 \
    --seed 42 \
    --exclude-overlap-file "$EVAL_PROMPTS" \
    --overlap-ngram-size 32
  date -Is > "$RUN_DIR/PROMPTS_DONE"
fi

if [[ ! -f "$RUN_DIR/SELF_DISTILLATION_DONE" ]]; then
  "$PYTHON" "$ROOT_DIR/tools/generate_gemma4_self_distillation.py" \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output "$SELF_DISTILLED" \
    --max-model-len 1024 \
    --max-new-tokens 256 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --generation-batch-size 512 \
    --temperature 0.6 \
    --seed 0 \
    --resume
  date -Is > "$RUN_DIR/SELF_DISTILLATION_DONE"
fi

if [[ ! -f "$RUN_DIR/CACHE_DONE" ]]; then
  "$PYTHON" "$ROOT_DIR/tools/generate_dflash_data.py" \
    --target_backend vllm \
    --target_model_name_or_path "$MODEL" \
    --draft_model_config_path "$ROOT_DIR/configs/gemma4_dflare.json" \
    --train_data_path "$SELF_DISTILLED" \
    --output_dir "$CACHE" \
    --chat_template_type gemma4 \
    --model_max_length 1024 \
    --min_loss_tokens 16 \
    --batch_size 1 \
    --num_proc 8 \
    --shard_size 100 \
    --gpu_memory_utilization 0.8 \
    --max_num_seqs 1 \
    --max_cache_tokens 896 \
    --resume
  date -Is > "$RUN_DIR/CACHE_DONE"
fi

date -Is > "$RUN_DIR/PIPELINE_DONE"
