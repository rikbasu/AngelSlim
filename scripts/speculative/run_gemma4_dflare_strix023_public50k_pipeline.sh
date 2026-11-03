#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/scratch/rbasu/AngelSlim}
VLLM_ROOT=${VLLM_ROOT:-/scratch/rbasu/vllm-nightly-dflare}
PYTHON=${PYTHON:-/scratch/rbasu/rocm-scripts/venv-gemma-dflare/bin/python}
MODEL=${MODEL:-/proj/rdi/staff/rbasu/nobkup/models/gemma4/gemma-4-E4B-it}
DATA=${DATA:-/proj/rdi/staff/rbasu/nobkup/datasets/gemma4-dflare/dflare_public_50k_mi300_distilled.jsonl}
CACHE_DIR=${CACHE_DIR:-/proj/rdi/staff/rbasu/nobkup/datasets/gemma4-dflare/offline-cache-public50k-vllm023}
JOB_DIR=${JOB_DIR:-/scratch/rbasu/dflare-jobs/public50k-vllm023}
REMOTE=${REMOTE:-rbasu@10.23.233.219}
REMOTE_CACHE=${REMOTE_CACHE:-/home/rbasu/data/offline-cache-public50k-vllm023}
REMOTE_INITIAL=${REMOTE_INITIAL:-/home/rbasu/outputs/dflare-strix023-cache-aligned-10k-1ep}
REMOTE_OUTPUT=${REMOTE_OUTPUT:-/home/rbasu/outputs/dflare-strix023-cache-aligned-public50k-1ep}
REMOTE_JOB=${REMOTE_JOB:-/home/rbasu/jobs/dflare-strix023-public50k-alignment}
LOCAL_OUTPUT=${LOCAL_OUTPUT:-/proj/rdi/staff/rbasu/nobkup/models/gemma4/dflare-strix023-cache-aligned-public50k-1ep}
VLLM_OUTPUT=${VLLM_OUTPUT:-${LOCAL_OUTPUT}-vllm-reference}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-/scratch/rbasu/rocm-scripts/experiments/gemma4_e4b_dflare}
DATASET=${DATASET:-/proj/rdi/staff/rbasu/nobkup/datasets/gemma4-dflare/public_mixed_60_eval_prompts.jsonl}

mkdir -p "$JOB_DIR" "$EXPERIMENT_DIR"
exec 9>"$JOB_DIR/pipeline.lock"
if ! flock -n 9; then
  echo "The exact-vLLM 50K pipeline is already running"
  exit 0
fi
if [[ -f "$JOB_DIR/PIPELINE_DONE" ]]; then
  echo "The exact-vLLM 50K pipeline already completed"
  exit 0
fi

exec >>"$JOB_DIR/pipeline.log" 2>&1
echo "$(date -Is) pipeline start or resume"

export PYTHONPATH="$VLLM_ROOT:/scratch/rbasu/rocm-scripts/venv-gemma-dflare/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi:$ROOT_DIR:${PYTHONPATH:-}"
export VLLM_TARGET_DEVICE=rocm
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=""
export HIP_VISIBLE_DEVICES=0

if [[ ! -f "$JOB_DIR/CACHE_DONE" ]]; then
  mkdir -p "$CACHE_DIR"
  while true; do
    echo "$(date -Is) generating exact Strix vLLM 0.23 cache"
    if "$PYTHON" "$ROOT_DIR/tools/generate_dflash_data.py" \
      --target_backend vllm \
      --target_model_name_or_path "$MODEL" \
      --draft_model_config_path "$ROOT_DIR/configs/gemma4_dflare.json" \
      --train_data_path "$DATA" \
      --output_dir "$CACHE_DIR" \
      --chat_template_type gemma4 \
      --model_max_length 1024 \
      --min_loss_tokens 16 \
      --batch_size 1 \
      --num_proc 8 \
      --shard_size 100 \
      --gpu_memory_utilization 0.8 \
      --max_num_seqs 1 \
      --resume
    then
      date -Is > "$JOB_DIR/CACHE_DONE"
      break
    fi
    echo "$(date -Is) cache generator stopped; retrying in 60 seconds"
    sleep 60
  done
fi

if [[ ! -f "$JOB_DIR/TRANSFER_DONE" ]]; then
  while ! rsync -a --partial --info=stats2 "$CACHE_DIR/" "$REMOTE:$REMOTE_CACHE/"; do
    echo "$(date -Is) cache transfer interrupted; retrying in 60 seconds"
    sleep 60
  done
  date -Is > "$JOB_DIR/TRANSFER_DONE"
fi

launch_remote_training() {
  ssh -o BatchMode=yes "$REMOTE" "
    set -e
    mkdir -p '$REMOTE_JOB'
    if [[ -f '$REMOTE_JOB/TRAINING_DONE' ]]; then
      exit 0
    fi
    if [[ -f '$REMOTE_JOB/training.pid' ]] &&
       kill -0 \"\$(< '$REMOTE_JOB/training.pid')\" 2>/dev/null; then
      exit 0
    fi
    nohup env \
      CACHE_DIR='$REMOTE_CACHE' \
      INITIAL_DRAFT='$REMOTE_INITIAL' \
      OUTPUT_DIR='$REMOTE_OUTPUT' \
      JOB_DIR='$REMOTE_JOB' \
      /home/rbasu/AngelSlim/scripts/speculative/run_gemma4_dflare_strix023_cache_mi300.sh \
      > '$REMOTE_JOB/training.log' 2>&1 < /dev/null &
    echo \$! > '$REMOTE_JOB/training.pid'
  "
}

if [[ ! -f "$JOB_DIR/TRAINING_DONE" ]]; then
  while true; do
    if ! ssh -o BatchMode=yes "$REMOTE" \
      "test -f '$REMOTE_JOB/TRAINING_DONE'"; then
      launch_remote_training || true
      sleep 60
      continue
    fi
    date -Is > "$JOB_DIR/TRAINING_DONE"
    break
  done
fi

if [[ ! -f "$JOB_DIR/CHECKPOINT_TRANSFER_DONE" ]]; then
  mkdir -p "$LOCAL_OUTPUT"
  while ! rsync -a --partial --exclude='checkpoint-*' \
    "$REMOTE:$REMOTE_OUTPUT/" "$LOCAL_OUTPUT/"; do
    echo "$(date -Is) checkpoint transfer interrupted; retrying in 60 seconds"
    sleep 60
  done
  date -Is > "$JOB_DIR/CHECKPOINT_TRANSFER_DONE"
fi

if [[ ! -f "$JOB_DIR/CONVERSION_DONE" ]]; then
  if [[ ! -f "$VLLM_OUTPUT/model.safetensors" ]]; then
    "$PYTHON" "$VLLM_ROOT/tools/convert_angelslim_dflare.py" \
      "$LOCAL_OUTPUT" "$VLLM_OUTPUT" \
      --block-size 16 \
      --num-speculative-tokens 15 \
      --reference-attention
  fi
  date -Is > "$JOB_DIR/CONVERSION_DONE"
fi

export PYTHONPATH="$VLLM_ROOT:/scratch/rbasu/rocm-scripts/venv-gemma-dflare/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi:$ROOT_DIR"

for depth in 3 7 15; do
  output="$EXPERIMENT_DIR/benchmark_public60_dflare50k_strix023cache_full50k_k${depth}_eager.jsonl"
  marker="$JOB_DIR/BENCHMARK_K${depth}_DONE"
  temp="$JOB_DIR/benchmark_k${depth}.jsonl"
  if [[ -f "$marker" ]]; then
    continue
  fi
  if [[ -f "$temp" ]]; then
    mv "$temp" "$temp.incomplete.$(date +%s)"
  fi
  "$PYTHON" "$ROOT_DIR/tools/vllm_spec_benchmark.py" \
    --dataset "$DATASET" \
    --target_model "$MODEL" \
    --draft_model "$VLLM_OUTPUT" \
    --method dflare \
    --num_spec_tokens "$depth" \
    --num_prompts 60 \
    --output_len 64 \
    --enforce_eager \
    --profile_components \
    --output_file "$temp"
  mv "$temp" "$output"
  date -Is > "$marker"
done

date -Is > "$JOB_DIR/PIPELINE_DONE"
echo "$(date -Is) pipeline complete"
