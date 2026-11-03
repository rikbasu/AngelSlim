# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
usage:
# 使用 benchmark 名称（自动从 dataset/ 目录加载）
python3 vllm_spec_benchmark.py \
    --target_model "$MODEL_DIR" \
    --draft_model "$EAGLE_DIR" \
    --dataset "gsm8k" \
    --output_file "$OUTPUT_FILE" \
    --method eagle3 \
    --output_len 1024 \
    --max_num_seqs "$BATCH_SIZE"

# 使用本地文件路径
python3 vllm_spec_benchmark.py \
    --target_model "$MODEL_DIR" \
    --draft_model "$EAGLE_DIR" \
    --dataset "/path/to/local/question.jsonl" \
    --output_file "$OUTPUT_FILE" \
    --method eagle3 \
    --output_len 1024 \
    --max_num_seqs "$BATCH_SIZE"

# 多个数据集（逗号分隔，支持混合使用）
python3 vllm_spec_benchmark.py \
    --target_model "$MODEL_DIR" \
    --draft_model "$EAGLE_DIR" \
    --dataset "mt_bench,gsm8k,/path/to/local/question.jsonl" \
    --output_file "$OUTPUT_FILE" \
    --method eagle3 \
    --output_len 1024 \
    --max_num_seqs "$BATCH_SIZE"

# 可重复的深度 / 执行模式扫描
python3 vllm_spec_benchmark.py \
    --target_model "$MODEL_DIR" \
    --draft_model "$EAGLE_DIR" \
    --dataset "gsm8k" \
    --output_file "$OUTPUT_FILE" \
    --method eagle3 \
    --depth_sweep \
    --execution_modes eager,cudagraph \
    --warmup_repetitions 1 \
    --repetitions 5 \
    --profile_components
"""

import argparse
import gc
import json
import math
import os
import statistics
import time
from datetime import datetime
from statistics import NormalDist


DEFAULT_SWEEP_DEPTHS = (1, 3, 7, 15)


def _runtime_dependencies():
    """Import GPU/runtime dependencies only when a benchmark is executed."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    return AutoTokenizer, LLM, SamplingParams, TokensPrompt


def _extract_text_content(content):
    """
    从 content 字段中提取纯文本内容。

    兼容两种格式：
    - 字符串格式: "xxx"
    - 列表格式: [{"type": "text", "text": "xxx"}, ...]

    Args:
        content: 字符串或列表格式的 content

    Returns:
        提取出的纯文本字符串
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 拼接所有 type=text 的文本片段
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return str(content)


def _extract_prompts(ds, num_prompts):
    """
    从数据集中提取 prompt 列表，自动识别数据格式。

    支持的格式：
    1. turns 格式: {"turns": ["xxx"], "reference": ["xxx"]}
       - 取第一个 turn 作为 prompt，返回纯文本列表
    2. conversations 格式:
       {"conversations": [
          {"role": "system", "content": "xxx"},
          {"role": "user", "content": "xxx"},
          {"role": "assistant", ...}]
       }
       {"conversations": [{"role": "user", "content": [{"type": "text", "text": "xxx"}]}, ...]}
       - 提取 system + 第一个 user 消息，构建完整的 messages 列表

    Args:
        ds: HuggingFace Dataset 对象
        num_prompts: 最多提取的 prompt 数量

    Returns:
        prompts: 纯文本 prompt 列表（turns 格式）或 messages 列表的列表（conversations 格式）
        format_type: 数据格式类型，"turns" 或 "conversations"
    """
    columns = ds.column_names

    if "turns" in columns:
        # turns 格式：取每条数据的第一个 turn
        return [q[0] for q in ds["turns"][:num_prompts]], "turns"
    elif "prompt_token_ids" in columns:
        # Pre-tokenized prefixes are useful for train/runtime parity checks:
        # preserve exact cached token boundaries without decode/re-tokenize drift.
        return [
            [int(token_id) for token_id in token_ids]
            for token_ids in ds["prompt_token_ids"][:num_prompts]
        ], "token_ids"
    elif "conversations" in columns:
        # conversations 格式：提取 system prompt + 第一个 user 消息，构建完整 messages
        prompts = []
        for item in ds[:num_prompts]["conversations"]:
            messages = []
            for msg in item:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    messages.append({"role": "system", "content": _extract_text_content(content)})
                elif role == "user":
                    messages.append({"role": "user", "content": _extract_text_content(content)})
                    break  # 只取到第一个 user 消息
            if messages and messages[-1]["role"] == "user":
                prompts.append(messages)
        return prompts, "conversations"
    elif "messages" in columns:
        # messages 格式（sharegpt_gpt4_qwen 等）：与 conversations 格式类似
        prompts = []
        for item in ds[:num_prompts]["messages"]:
            messages = []
            for msg in item:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    messages.append({"role": "system", "content": _extract_text_content(content)})
                elif role == "user":
                    messages.append({"role": "user", "content": _extract_text_content(content)})
                    break
            if messages and messages[-1]["role"] == "user":
                prompts.append(messages)
        return prompts, "conversations"
    else:
        raise ValueError(
            f"Unsupported dataset format. Expected 'turns', 'conversations' or 'messages' column, "
            f"but found: {columns}"
        )


def load_benchmark_dataset(dataset, num_prompts, tokenizer, dataset_base_dir):
    """
    Load dataset based on dataset name or local file path

    Args:
        dataset: Benchmark name (folder name under dataset directory) or local JSONL file path.
                 If the path exists as a file, it will be loaded directly;
                 otherwise it is treated as a benchmark name under dataset_base_dir.
        num_prompts: Number of prompts to load
        tokenizer: Tokenizer for encoding prompts
        dataset_base_dir: Base directory containing all benchmark datasets

    Returns:
        prompt_ids: List of tokenized prompt ids
        prompts: List of original prompt texts
    """

    # 判断是本地文件路径还是 benchmark 名称
    if os.path.exists(dataset):
        dataset_file = dataset
        print(f"Loading dataset from local path: {dataset_file}")
    else:
        dataset_file = os.path.join(dataset_base_dir, dataset, "question.jsonl")

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # Load dataset lazily so CPU-only tooling/tests can import this module.
    from datasets import load_dataset

    ds = load_dataset("json", data_files=dataset_file)["train"]
    prompts, format_type = _extract_prompts(ds, num_prompts)

    # Tokenize prompts
    if format_type == "token_ids":
        prompt_ids = prompts
        prompts_text = [f"<{len(token_ids)} pre-tokenized tokens>" for token_ids in prompt_ids]
    elif format_type == "conversations":
        # conversations/messages 格式：使用 apply_chat_template 构建包含 system prompt 的完整 prompt
        prompt_ids = []
        for messages in prompts:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            prompt_ids.append(token_ids)
        # 将 messages 列表转为可读文本用于日志
        prompts_text = [
            next(
                (msg["content"] for msg in msgs if msg["role"] == "user"),
                "",
            )
            for msgs in prompts
        ]
    else:
        # turns 格式：直接编码纯文本
        # add_special_tokens is False to avoid adding bos twice when using chat templates
        prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
        prompts_text = prompts

    print(f"Loaded {len(prompt_ids)} prompts from: {dataset_file} (dataset={dataset})")
    print(
        "Average prompt length: " f"{sum(len(p) for p in prompt_ids) / len(prompt_ids):.2f} tokens"
    )

    return prompt_ids, prompts_text


def save_stats_to_jsonl(stats, output_file):
    """
    Save benchmark statistics to a jsonl file

    Args:
        stats: Dictionary containing benchmark statistics
        output_file: Path to the output jsonl file
    """
    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )

    # Append stats to jsonl file
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Stats saved to: {output_file}")


def _parse_positive_int_csv(value, option_name):
    """Parse a comma-separated list of unique positive integers."""
    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raw_values = str(value).split(",")

    result = []
    for raw_value in raw_values:
        try:
            parsed = int(str(raw_value).strip())
        except ValueError as exc:
            raise ValueError(f"{option_name} must contain integers: {value!r}") from exc
        if parsed <= 0:
            raise ValueError(f"{option_name} values must be positive: {parsed}")
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise ValueError(f"{option_name} cannot be empty")
    return result


def _normalize_execution_modes(value, enforce_eager):
    """Return stable mode names and their vLLM ``enforce_eager`` values."""
    if value is None:
        return [("eager" if enforce_eager else "cudagraph", enforce_eager)]

    aliases = {
        "eager": ("eager", True),
        "non-eager": ("cudagraph", False),
        "non_eager": ("cudagraph", False),
        "cudagraph": ("cudagraph", False),
        "cuda-graph": ("cudagraph", False),
        "graph": ("cudagraph", False),
    }
    result = []
    for raw_mode in str(value).split(","):
        mode = raw_mode.strip().lower()
        if mode not in aliases:
            raise ValueError(
                "--execution_modes supports eager and cudagraph/non-eager; "
                f"got {raw_mode!r}"
            )
        normalized = aliases[mode]
        if normalized not in result:
            result.append(normalized)
    if not result:
        raise ValueError("--execution_modes cannot be empty")
    return result


def resolve_sweep_configurations(args):
    """Resolve depth/mode combinations without requiring vLLM."""
    if args.sweep_depths:
        depths = _parse_positive_int_csv(args.sweep_depths, "--sweep_depths")
    elif args.depth_sweep:
        depths = list(DEFAULT_SWEEP_DEPTHS)
    else:
        depths = [args.num_spec_tokens]

    modes = _normalize_execution_modes(args.execution_modes, args.enforce_eager)
    return [
        {
            "num_spec_tokens": depth,
            "execution_mode": mode,
            "enforce_eager": enforce_eager,
        }
        for mode, enforce_eager in modes
        for depth in depths
    ]


def parse_args(argv=None):
    try:
        from vllm.utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser()
    except ImportError:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Benchmark name or local JSONL file path. Supports comma-separated values "
        "for multiple datasets (e.g. 'mt_bench,gsm8k,/path/to/local.jsonl'). "
        "If the value is an existing file path, it will be loaded directly; "
        "otherwise it is treated as a benchmark name under the dataset/ directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=80,
        help="Number of prompts to load from the dataset",
    )

    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="eagle3",
        choices=["ngram", "eagle", "eagle3", "dflare", "mtp", "ar"],
    )
    parser.add_argument("--num_spec_tokens", type=int, default=2)
    parser.add_argument(
        "--depth_sweep",
        action="store_true",
        help="Sweep speculative depths 1,3,7,15. The default remains a single run.",
    )
    parser.add_argument(
        "--sweep_depths",
        type=str,
        default=None,
        help="Optional comma-separated speculative depths; implies a depth sweep.",
    )
    parser.add_argument(
        "--execution_modes",
        type=str,
        default=None,
        help="Comma-separated eager and/or cudagraph (alias: non-eager). "
        "By default, preserve --enforce_eager behavior.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Measured repetitions for each dataset/depth/mode combination.",
    )
    parser.add_argument(
        "--warmup_repetitions",
        type=int,
        default=0,
        help="Unrecorded warmup repetitions for each combination.",
    )
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.95,
        help="Two-sided normal confidence level for repetition summaries.",
    )
    parser.add_argument(
        "--profile_components",
        action="store_true",
        help="Record draft/target/runtime timing metrics exposed by vLLM. "
        "Does not enable an invasive framework profiler.",
    )
    parser.add_argument("--prompt_lookup_max", type=int, default=5)
    parser.add_argument("--prompt_lookup_min", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--use_local_argmax_reduction",
        action="store_true",
        help=(
            "Ask vLLM to select draft argmax IDs without materializing full-vocab "
            "logits. Recommended for reduced-vocabulary greedy DFlare checkpoints."
        ),
    )
    parser.add_argument(
        "--draft_attention_backend",
        default=None,
        help=(
            "Optional vLLM speculative attention backend override, for example "
            "FLASH_ATTN for DFlash/DFlare non-causal block attention."
        ),
    )
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--enable_chunked_prefill", action="store_true")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--print_output", action="store_true")
    parser.add_argument(
        "--save_token_ids",
        action="store_true",
        help="Include generated token IDs in repetition records for parity diagnostics.",
    )
    parser.add_argument("--output_len", type=int, default=1024)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument(
        "--output_file",
        type=str,
        default="benchmark_stats.jsonl",
        help="Path to save benchmark statistics in jsonl format",
    )
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft_model", type=str, default="AngelSlim/Qwen3-1.7B_eagle3")
    args = parser.parse_args(argv)
    if args.num_spec_tokens <= 0:
        parser.error("--num_spec_tokens must be positive")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.warmup_repetitions < 0:
        parser.error("--warmup_repetitions cannot be negative")
    if not 0 < args.confidence_level < 1:
        parser.error("--confidence_level must be between 0 and 1")
    try:
        resolve_sweep_configurations(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _add_metric_value(existing, value):
    if existing is None:
        return value
    if isinstance(existing, list) and isinstance(value, list):
        size = max(len(existing), len(value))
        return [
            (existing[index] if index < len(existing) else 0)
            + (value[index] if index < len(value) else 0)
            for index in range(size)
        ]
    if isinstance(existing, (int, float)) and isinstance(value, (int, float)):
        return existing + value
    return value


def snapshot_metrics(metrics):
    """Convert vLLM metric objects into a version-tolerant numeric snapshot."""
    snapshot = {}
    for metric in metrics or []:
        name = getattr(metric, "name", None)
        if not name:
            continue
        values = getattr(metric, "values", None)
        value = getattr(metric, "value", None)
        if values is not None:
            numeric_value = [float(item) for item in values]
        elif isinstance(value, (int, float)):
            numeric_value = float(value)
        else:
            # Prometheus histogram-like readers commonly expose ``sum``.
            metric_sum = getattr(metric, "sum", None)
            if not isinstance(metric_sum, (int, float)):
                continue
            numeric_value = float(metric_sum)
        snapshot[name] = _add_metric_value(snapshot.get(name), numeric_value)
    return snapshot


def diff_metric_snapshots(before, after):
    """Diff cumulative snapshots, while tolerating reset-on-read metrics."""
    result = {}
    for name, after_value in after.items():
        before_value = before.get(name)
        if isinstance(after_value, list):
            before_values = before_value if isinstance(before_value, list) else []
            values = []
            for index, value in enumerate(after_value):
                previous = before_values[index] if index < len(before_values) else 0
                values.append(value - previous if value >= previous else value)
            result[name] = values
        elif isinstance(after_value, (int, float)):
            previous = before_value if isinstance(before_value, (int, float)) else 0
            result[name] = after_value - previous if after_value >= previous else after_value
    return result


def _read_metric_snapshot(llm):
    try:
        return snapshot_metrics(llm.get_metrics()), True
    except (AssertionError, AttributeError, NotImplementedError):
        return {}, False


def _metric_scalar(metrics, name):
    value = metrics.get(name, 0)
    return value if isinstance(value, (int, float)) else 0


def _percentile(values, percentile):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summarize_values(values):
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not numeric_values:
        return None
    return {
        "count": len(numeric_values),
        "mean": round(statistics.fmean(numeric_values), 6),
        "min": round(min(numeric_values), 6),
        "p50": round(_percentile(numeric_values, 0.5), 6),
        "p95": round(_percentile(numeric_values, 0.95), 6),
        "max": round(max(numeric_values), 6),
    }


def extract_request_timings(outputs):
    """Summarize RequestMetrics fields exposed by the installed vLLM."""
    samples = {
        "e2e_seconds": [],
        "time_to_first_token_seconds": [],
        "queue_seconds": [],
        "prefill_seconds": [],
        "decode_after_first_token_seconds": [],
        "scheduler_seconds": [],
        "model_forward_seconds": [],
        "model_execute_seconds": [],
    }
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            continue

        def number(name):
            value = getattr(metrics, name, None)
            return float(value) if isinstance(value, (int, float)) else None

        arrival = number("arrival_time")
        scheduled = number("first_scheduled_time")
        first_token = number("first_token_time")
        finished = number("finished_time")
        queue = number("time_in_queue")

        derived = {
            "e2e_seconds": finished - arrival
            if finished is not None and arrival is not None
            else None,
            "time_to_first_token_seconds": first_token - arrival
            if first_token is not None and arrival is not None
            else None,
            "queue_seconds": queue
            if queue is not None
            else (
                scheduled - arrival
                if scheduled is not None and arrival is not None
                else None
            ),
            "prefill_seconds": first_token - scheduled
            if first_token is not None and scheduled is not None
            else None,
            "decode_after_first_token_seconds": finished - first_token
            if finished is not None and first_token is not None
            else None,
            "scheduler_seconds": number("scheduler_time"),
            "model_forward_seconds": number("model_forward_time"),
            "model_execute_seconds": number("model_execute_time"),
        }
        for name, value in derived.items():
            if value is not None and value >= 0:
                samples[name].append(value)

    return {
        name: summary
        for name, values in samples.items()
        if (summary := _summarize_values(values)) is not None
    }


def extract_component_timings(metric_values):
    """Group timing counters exposed by vLLM without enabling a profiler."""
    components = {"draft": {}, "target": {}, "runtime": {}}
    timing_words = ("time", "latency", "duration", "seconds")
    for name, value in metric_values.items():
        lower_name = name.lower()
        if not any(word in lower_name for word in timing_words):
            continue
        if isinstance(value, list):
            numeric_value = sum(value)
        elif isinstance(value, (int, float)):
            numeric_value = value
        else:
            continue
        if any(word in lower_name for word in ("target", "verify", "verification")):
            component = "target"
        elif "draft" in lower_name:
            component = "draft"
        else:
            component = "runtime"
        components[component][name] = round(float(numeric_value), 6)
    return components


def confidence_statistics(values, confidence_level=0.95):
    """Return mean, spread, and a two-sided normal confidence interval."""
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not numeric_values:
        return None
    count = len(numeric_values)
    mean = statistics.fmean(numeric_values)
    standard_deviation = statistics.stdev(numeric_values) if count > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(count)
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    margin = z_score * standard_error
    return {
        "n": count,
        "mean": round(mean, 6),
        "standard_deviation": round(standard_deviation, 6),
        "standard_error": round(standard_error, 6),
        "confidence_level": confidence_level,
        "ci_lower": round(mean - margin, 6),
        "ci_upper": round(mean + margin, 6),
    }


def _nested_numeric_values(records, path):
    values = []
    for record in records:
        current = record
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)):
            values.append(current)
    return values


def calculate_repetition_summary(records, args):
    """Build a machine-readable confidence summary for repeated runs."""
    first = records[0]
    fields = (
        "latency_seconds",
        "output_throughput_tokens_per_sec",
        "request_throughput_per_sec",
        "avg_time_per_sample",
        "acceptance_rate",
        "mean_acceptance_length",
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
    )
    summary = {
        "record_type": "benchmark_summary",
        "timestamp": datetime.now().isoformat(),
        "dataset": first["dataset"],
        "model_dir": first["model_dir"],
        "eagle_dir": first.get("eagle_dir"),
        "draft_dir": first.get("draft_dir"),
        "method": first["method"],
        "num_spec_tokens": first["num_spec_tokens"],
        "execution_mode": first["execution_mode"],
        "enforce_eager": first["enforce_eager"],
        "repetitions": len(records),
        "confidence_method": "two-sided normal approximation",
        "confidence_level": args.confidence_level,
        "statistics": {
            field: confidence_statistics(
                [record[field] for record in records if field in record],
                args.confidence_level,
            )
            for field in fields
        },
        "timing_statistics": {"overall": {}, "draft": {}, "target": {}, "runtime": {}},
    }
    summary["statistics"] = {
        name: value for name, value in summary["statistics"].items() if value is not None
    }

    timing_paths = set()
    for record in records:
        for component, component_values in record.get("timing", {}).items():
            if not isinstance(component_values, dict):
                continue
            for name, value in component_values.items():
                if isinstance(value, (int, float)):
                    timing_paths.add((component, name))
                elif isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
                    timing_paths.add((component, name, "mean"))
    for path in sorted(timing_paths):
        values = _nested_numeric_values(records, ("timing",) + path)
        statistic = confidence_statistics(values, args.confidence_level)
        if statistic is not None:
            key = path[1] if len(path) == 2 else f"{path[1]}_mean"
            summary["timing_statistics"][path[0]][key] = statistic
    return summary


def run_benchmark(
    args,
    dataset_name,
    llm,
    tokenizer,
    sampling_params,
    *,
    tokens_prompt_cls=None,
    prompt_ids=None,
    num_spec_tokens=None,
    execution_mode=None,
    enforce_eager=None,
    repetition=1,
    warmup=False,
):
    """
    Run benchmark for a single dataset

    Args:
        args: Command line arguments
        dataset_name: Benchmark name or local file path
        llm: Initialized LLM model
        tokenizer: Tokenizer instance
        sampling_params: Sampling parameters

    Returns:
        stats: Dictionary containing all benchmark statistics
    """
    print("\n" + "=" * 80)
    print(f"Running benchmark: {dataset_name}")
    print("=" * 80)

    if prompt_ids is None:
        current_file_path = os.path.abspath(__file__)
        dataset_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(current_file_path)), "dataset"
        )
        prompt_ids, _ = load_benchmark_dataset(
            dataset=dataset_name,
            num_prompts=args.num_prompts,
            tokenizer=tokenizer,
            dataset_base_dir=dataset_base_dir,
        )
    if tokens_prompt_cls is None:
        from vllm.inputs import TokensPrompt as tokens_prompt_cls

    depth = num_spec_tokens if num_spec_tokens is not None else args.num_spec_tokens
    eager = args.enforce_eager if enforce_eager is None else enforce_eager
    mode = execution_mode or ("eager" if eager else "cudagraph")

    # Generate outputs
    before_metrics, before_metrics_available = _read_metric_snapshot(llm)
    tic = time.perf_counter()
    outputs = llm.generate(
        [tokens_prompt_cls(prompt_token_ids=x) for x in prompt_ids],
        sampling_params=sampling_params,
    )
    latency = time.perf_counter() - tic

    # Print generated text if requested
    if args.print_output:
        for i, output in enumerate(outputs):
            print("-" * 50)
            print(f"Prompt {i + 1}: {output.prompt}")
            print(f"Generated text: {output.outputs[0].text}")
            print("-" * 50)

    # Get and process metrics. Throughput/timing remain valid if the engine
    # version does not expose metric-reader counters.
    after_metrics, after_metrics_available = _read_metric_snapshot(llm)
    metric_values = diff_metric_snapshots(before_metrics, after_metrics)
    metrics_available = before_metrics_available and after_metrics_available
    if not metrics_available:
        print("Metric-reader counters are not supported by this vLLM engine.")

    total_num_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    num_drafts = _metric_scalar(metric_values, "vllm:spec_decode_num_drafts")
    num_draft_tokens = _metric_scalar(
        metric_values, "vllm:spec_decode_num_draft_tokens"
    )
    num_accepted_tokens = _metric_scalar(
        metric_values, "vllm:spec_decode_num_accepted_tokens"
    )
    per_position = metric_values.get(
        "vllm:spec_decode_num_accepted_tokens_per_pos", []
    )
    acceptance_counts = [0.0] * max(depth, len(per_position))
    for position, value in enumerate(per_position):
        acceptance_counts[position] += value

    total_num_input_tokens = sum(len(p) for p in prompt_ids)
    num_prompts = len(prompt_ids)
    output_throughput = total_num_output_tokens / latency
    request_throughput = num_prompts / latency
    avg_time_per_sample = latency / num_prompts if num_prompts > 0 else 0
    avg_input_tokens = total_num_input_tokens / num_prompts if num_prompts > 0 else 0
    avg_output_tokens = total_num_output_tokens / num_prompts if num_prompts > 0 else 0
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    overall_acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0
    )

    # Calculate acceptance rate at each position
    acceptance_rates = {}
    for i in range(len(acceptance_counts)):
        position_acceptance_rate = (
            acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        )
        acceptance_rates[f"acceptance_rate_pos_{i}"] = round(
            position_acceptance_rate, 4
        )

    # Prepare statistics dictionary
    component_timings = (
        extract_component_timings(metric_values)
        if args.profile_components
        else {"draft": {}, "target": {}, "runtime": {}}
    )
    stats = {
        "record_type": "benchmark_repetition",
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "model_dir": args.target_model,
        "eagle_dir": args.draft_model if args.method in ["eagle", "eagle3"] else None,
        "draft_dir": args.draft_model if args.method == "dflare" else None,
        "method": args.method,
        "num_spec_tokens": depth,
        "execution_mode": mode,
        "enforce_eager": eager,
        "repetition": repetition,
        "warmup": warmup,
        "num_prompts": num_prompts,
        "output_len": args.output_len,
        "temperature": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "total_num_input_tokens": total_num_input_tokens,
        "total_num_output_tokens": total_num_output_tokens,
        "latency_seconds": round(latency, 2),
        "output_throughput_tokens_per_sec": round(output_throughput, 2),
        "request_throughput_per_sec": round(request_throughput, 2),
        "avg_time_per_sample": round(avg_time_per_sample, 4),
        "avg_input_tokens": round(avg_input_tokens, 2),
        "avg_output_tokens": round(avg_output_tokens, 2),
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "acceptance_rate": round(overall_acceptance_rate, 6),
        "mean_acceptance_length": round(acceptance_length, 4),
        "runtime_metrics_available": metrics_available,
        "timing": {
            "overall": {
                "wall_seconds": round(latency, 6),
                **extract_request_timings(outputs),
            },
            **component_timings,
        },
        **acceptance_rates,
    }
    if getattr(args, "save_token_ids", False):
        stats["generated_token_ids"] = [
            list(output.outputs[0].token_ids) for output in outputs
        ]

    # Print statistics
    print("-" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"total_num_input_tokens: {total_num_input_tokens}")
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"latency: {latency:.2f} s")
    print(f"output_throughput: {output_throughput:.2f} tokens/s")
    print(f"request_throughput: {request_throughput:.2f} requests/s")
    print(f"avg_time_per_sample: {avg_time_per_sample:.4f} s")
    print(f"avg_input_tokens: {avg_input_tokens:.2f}")
    print(f"avg_output_tokens: {avg_output_tokens:.2f}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    print(f"acceptance rate: {overall_acceptance_rate:.4f}")
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # Print acceptance rate at each token position
    for i in range(len(acceptance_counts)):
        position_acceptance_rate = (
            acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        )
        print(f"acceptance at token {i}: {position_acceptance_rate:.2f}")

    return stats


def calculate_average_stats(all_stats, args):
    """
    Calculate average statistics from multiple benchmark results

    Args:
        all_stats: List of statistics dictionaries from all benchmarks
        args: Command line arguments

    Returns:
        avg_stats: Dictionary containing average statistics
    """
    # Define numeric fields to calculate average
    numeric_fields = [
        "total_num_input_tokens",
        "total_num_output_tokens",
        "latency_seconds",
        "output_throughput_tokens_per_sec",
        "request_throughput_per_sec",
        "avg_time_per_sample",
        "avg_input_tokens",
        "avg_output_tokens",
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "mean_acceptance_length",
    ]

    # Add acceptance rate fields
    if all_stats[0].get("acceptance_rate_pos_0") is not None:
        for i in range(args.num_spec_tokens):
            numeric_fields.append(f"acceptance_rate_pos_{i}")

    # Initialize average stats dictionary
    avg_stats = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "AVERAGE",
        "model_dir": args.target_model,
        "eagle_dir": args.draft_model if args.method in ["eagle", "eagle3"] else None,
        "draft_dir": args.draft_model if args.method == "dflare" else None,
        "method": args.method,
        "num_spec_tokens": args.num_spec_tokens,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "temperature": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_benchmarks": len(all_stats),
    }

    # Calculate average for each numeric field
    for field in numeric_fields:
        values = [s[field] for s in all_stats if field in s]
        if values:
            avg_value = sum(values) / len(values)
            avg_stats[field] = (
                round(avg_value, 4) if isinstance(values[0], float) else int(avg_value)
            )

    return avg_stats


def print_average_statistics(avg_stats):
    """
    Print average statistics in a formatted way

    Args:
        avg_stats: Dictionary containing average statistics
    """
    print("\n" + "=" * 80)
    print("AVERAGE STATISTICS")
    print("=" * 80)
    print(f"Number of benchmarks: {avg_stats['num_benchmarks']}")
    print(
        "Average output throughput: "
        f"{avg_stats['output_throughput_tokens_per_sec']:.2f} tokens/s"
    )
    print(f"Average acceptance length: {avg_stats['mean_acceptance_length']:.4f}")
    print("=" * 80)


def print_summary(all_stats, output_file):
    """
    Print summary of all benchmark results

    Args:
        all_stats: List of statistics dictionaries from all benchmarks
        output_file: Path to the output file
    """
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for stats in all_stats:
        print(f"{stats['dataset']}: " f"acceptance_length = {stats['mean_acceptance_length']:.2f}")
    print("=" * 80)
    print(f"\nAll statistics saved to: {output_file}")


def build_speculative_config(args, num_spec_tokens):
    """Build the vLLM speculative config for one sweep depth."""
    if args.method in ("eagle", "eagle3", "dflare"):
        eagle_dir = args.draft_model
        if args.method == "eagle" and eagle_dir is None:
            eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
        elif args.method == "eagle3" and eagle_dir is None:
            eagle_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        elif args.method == "dflare" and eagle_dir is None:
            raise ValueError("--draft_model is required for dflare")

        speculative_config = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": num_spec_tokens,
        }
        if getattr(args, "use_local_argmax_reduction", False):
            speculative_config["use_local_argmax_reduction"] = True
        if getattr(args, "draft_attention_backend", None):
            speculative_config["attention_backend"] = args.draft_attention_backend
    elif args.method == "ngram":
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": num_spec_tokens,
            "prompt_lookup_max": args.prompt_lookup_max,
            "prompt_lookup_min": args.prompt_lookup_min,
        }
    elif args.method == "mtp":
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": num_spec_tokens,
        }
    elif args.method == "ar":
        speculative_config = None
    else:
        raise ValueError(f"unknown method: {args.method}")
    return speculative_config


def create_llm(args, llm_cls, model_dir, configuration):
    """Create a fresh engine because depth and graph mode are startup options."""
    return llm_cls(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=configuration["enforce_eager"],
        gpu_memory_utilization=0.8,
        speculative_config=build_speculative_config(
            args, configuration["num_spec_tokens"]
        ),
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_cudagraph_capture_size=24,
    )


def shutdown_llm(llm):
    """Release an engine between sweep points when its vLLM version permits."""
    engine = getattr(llm, "llm_engine", None)
    for owner in (llm, engine):
        shutdown = getattr(owner, "shutdown", None)
        if callable(shutdown):
            shutdown()
            break


def main(args):
    # Set endpoint type for dataset processing
    args.endpoint_type = "openai-chat"

    model_dir = args.target_model or "meta-llama/Llama-3.1-8B-Instruct"
    AutoTokenizer, LLM, SamplingParams, TokensPrompt = _runtime_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.output_len,
    )

    # Parse dataset names (support multiple datasets separated by comma)
    dataset_names = [name.strip() for name in args.dataset.split(",") if name.strip()]
    current_file_path = os.path.abspath(__file__)
    dataset_base_dir = os.path.join(
        os.path.dirname(os.path.dirname(current_file_path)), "dataset"
    )
    prompt_ids_by_dataset = {}
    for dataset_name in dataset_names:
        prompt_ids_by_dataset[dataset_name], _ = load_benchmark_dataset(
            dataset=dataset_name,
            num_prompts=args.num_prompts,
            tokenizer=tokenizer,
            dataset_base_dir=dataset_base_dir,
        )

    configurations = resolve_sweep_configurations(args)
    orchestrated = bool(
        args.depth_sweep
        or args.sweep_depths
        or args.execution_modes
        or args.repetitions > 1
        or args.warmup_repetitions > 0
    )

    all_stats = []
    for configuration in configurations:
        print(
            "\nStarting configuration: "
            f"depth={configuration['num_spec_tokens']}, "
            f"mode={configuration['execution_mode']}"
        )
        llm = create_llm(args, LLM, model_dir, configuration)
        try:
            for dataset_name in dataset_names:
                common_kwargs = {
                    "args": args,
                    "dataset_name": dataset_name,
                    "llm": llm,
                    "tokenizer": tokenizer,
                    "sampling_params": sampling_params,
                    "tokens_prompt_cls": TokensPrompt,
                    "prompt_ids": prompt_ids_by_dataset[dataset_name],
                    **configuration,
                }
                for warmup_index in range(args.warmup_repetitions):
                    print(
                        f"Warmup {warmup_index + 1}/{args.warmup_repetitions}: "
                        f"{dataset_name}"
                    )
                    run_benchmark(
                        **common_kwargs,
                        repetition=warmup_index + 1,
                        warmup=True,
                    )

                repetition_stats = []
                for repetition in range(1, args.repetitions + 1):
                    stats = run_benchmark(
                        **common_kwargs,
                        repetition=repetition,
                    )
                    repetition_stats.append(stats)
                    all_stats.append(stats)
                    save_stats_to_jsonl(stats, args.output_file)

                if orchestrated:
                    summary = calculate_repetition_summary(repetition_stats, args)
                    save_stats_to_jsonl(summary, args.output_file)
        finally:
            shutdown_llm(llm)
            del llm
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass

    # Preserve the original cross-dataset average for the legacy single-run path.
    if not orchestrated and len(all_stats) > 1:
        avg_stats = calculate_average_stats(all_stats, args)
        save_stats_to_jsonl(avg_stats, args.output_file)
        print_average_statistics(avg_stats)

    # Print summary
    print_summary(all_stats, args.output_file)

    return all_stats


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
