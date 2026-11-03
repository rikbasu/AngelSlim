# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only tests for the vLLM speculative benchmark orchestration helpers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "vllm_spec_benchmark.py"
SPEC = importlib.util.spec_from_file_location("vllm_spec_benchmark", MODULE_PATH)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def _args(**overrides):
    values = {
        "num_spec_tokens": 2,
        "depth_sweep": False,
        "sweep_depths": None,
        "execution_modes": None,
        "enforce_eager": False,
        "confidence_level": 0.95,
        "print_output": False,
        "profile_components": True,
        "target_model": "target",
        "draft_model": "draft",
        "method": "eagle3",
        "num_prompts": 1,
        "output_len": 4,
        "temp": 0,
        "top_p": 1.0,
        "top_k": -1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_configuration_preserves_single_run():
    configurations = BENCHMARK.resolve_sweep_configurations(_args(enforce_eager=True))

    assert configurations == [
        {
            "num_spec_tokens": 2,
            "execution_mode": "eager",
            "enforce_eager": True,
        }
    ]


def test_depth_and_mode_sweep_is_repeatable_and_ordered():
    args = _args(
        depth_sweep=True,
        sweep_depths=None,
        execution_modes="eager,non-eager",
    )

    configurations = BENCHMARK.resolve_sweep_configurations(args)

    assert [item["num_spec_tokens"] for item in configurations] == [
        1,
        3,
        7,
        15,
        1,
        3,
        7,
        15,
    ]
    assert [item["execution_mode"] for item in configurations] == [
        "eager",
        "eager",
        "eager",
        "eager",
        "cudagraph",
        "cudagraph",
        "cudagraph",
        "cudagraph",
    ]


def test_custom_depths_are_deduplicated_and_validated():
    configurations = BENCHMARK.resolve_sweep_configurations(
        _args(sweep_depths="1, 3,3,7")
    )
    assert [item["num_spec_tokens"] for item in configurations] == [1, 3, 7]

    with pytest.raises(ValueError, match="positive"):
        BENCHMARK.resolve_sweep_configurations(_args(sweep_depths="1,0"))


def test_extract_prompts_preserves_pre_tokenized_prefixes():
    class TokenDataset:
        column_names = ["prompt_token_ids"]

        def __getitem__(self, key):
            assert key == "prompt_token_ids"
            return [[1, 2, 3], [4, 5]]

    prompts, format_type = BENCHMARK._extract_prompts(TokenDataset(), 1)

    assert prompts == [[1, 2, 3]]
    assert format_type == "token_ids"


def test_reduced_vocab_argmax_flag_reaches_speculative_config():
    config = BENCHMARK.build_speculative_config(
        _args(
            method="dflare",
            draft_model="reduced-draft",
            use_local_argmax_reduction=True,
            draft_attention_backend="FLASH_ATTN",
        ),
        3,
    )

    assert config["use_local_argmax_reduction"] is True
    assert config["attention_backend"] == "FLASH_ATTN"
    assert config["num_speculative_tokens"] == 3


def test_metric_snapshots_diff_counters_and_vectors():
    class Metric:
        def __init__(self, name, *, value=None, values=None):
            self.name = name
            self.value = value
            self.values = values

    before = BENCHMARK.snapshot_metrics(
        [
            Metric("drafts", value=10),
            Metric("positions", values=[8, 4, 2]),
        ]
    )
    after = BENCHMARK.snapshot_metrics(
        [
            Metric("drafts", value=14),
            Metric("positions", values=[11, 6, 2]),
        ]
    )

    assert BENCHMARK.diff_metric_snapshots(before, after) == {
        "drafts": 4.0,
        "positions": [3.0, 2.0, 0.0],
    }


def test_confidence_statistics_include_spread_and_interval():
    stats = BENCHMARK.confidence_statistics([10, 20, 30], 0.95)

    assert stats["n"] == 3
    assert stats["mean"] == 20
    assert stats["standard_deviation"] == 10
    assert stats["standard_error"] == pytest.approx(5.773503)
    assert stats["ci_lower"] < stats["mean"] < stats["ci_upper"]


def test_run_benchmark_records_acceptance_and_available_timings():
    class Metric:
        def __init__(self, name, *, value=None, values=None):
            self.name = name
            self.value = value
            self.values = values

    before = [
        Metric("vllm:spec_decode_num_drafts", value=10),
        Metric("vllm:spec_decode_num_draft_tokens", value=20),
        Metric("vllm:spec_decode_num_accepted_tokens", value=12),
        Metric(
            "vllm:spec_decode_num_accepted_tokens_per_pos",
            values=[8, 4, 2],
        ),
        Metric("vllm:spec_decode_draft_time_seconds", value=2),
        Metric("vllm:spec_decode_verification_time_seconds", value=3),
    ]
    after = [
        Metric("vllm:spec_decode_num_drafts", value=14),
        Metric("vllm:spec_decode_num_draft_tokens", value=25),
        Metric("vllm:spec_decode_num_accepted_tokens", value=15),
        Metric(
            "vllm:spec_decode_num_accepted_tokens_per_pos",
            values=[11, 6, 2],
        ),
        Metric("vllm:spec_decode_draft_time_seconds", value=2.4),
        Metric("vllm:spec_decode_verification_time_seconds", value=3.8),
    ]

    request_metrics = SimpleNamespace(
        arrival_time=1.0,
        first_scheduled_time=1.1,
        first_token_time=1.4,
        finished_time=2.0,
        time_in_queue=0.1,
        scheduler_time=0.05,
        model_forward_time=0.6,
        model_execute_time=0.7,
    )
    output = SimpleNamespace(
        prompt="prompt",
        metrics=request_metrics,
        outputs=[SimpleNamespace(token_ids=[1, 2, 3], text="result")],
    )

    class FakeLLM:
        def __init__(self):
            self.metric_reads = iter((before, after))

        def get_metrics(self):
            return next(self.metric_reads)

        def generate(self, prompts, sampling_params):
            assert prompts[0].prompt_token_ids == [10, 11]
            return [output]

    class FakeTokensPrompt:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    stats = BENCHMARK.run_benchmark(
        _args(num_spec_tokens=3),
        "cpu-fixture",
        FakeLLM(),
        tokenizer=None,
        sampling_params=object(),
        tokens_prompt_cls=FakeTokensPrompt,
        prompt_ids=[[10, 11]],
        num_spec_tokens=3,
        execution_mode="cudagraph",
        enforce_eager=False,
        repetition=2,
    )

    assert stats["record_type"] == "benchmark_repetition"
    assert stats["repetition"] == 2
    assert stats["num_drafts"] == 4
    assert stats["num_draft_tokens"] == 5
    assert stats["num_accepted_tokens"] == 3
    assert stats["acceptance_rate"] == 0.6
    assert stats["mean_acceptance_length"] == 1.75
    assert stats["timing"]["overall"]["time_to_first_token_seconds"]["mean"] == 0.4
    assert stats["timing"]["draft"]["vllm:spec_decode_draft_time_seconds"] == 0.4
    assert (
        stats["timing"]["target"][
            "vllm:spec_decode_verification_time_seconds"
        ]
        == 0.8
    )


def test_repetition_summary_has_confidence_and_component_sections():
    records = []
    for repetition, throughput in enumerate((100.0, 110.0, 120.0), start=1):
        records.append(
            {
                "dataset": "fixture",
                "model_dir": "target",
                "eagle_dir": "draft",
                "draft_dir": None,
                "method": "eagle3",
                "num_spec_tokens": 3,
                "execution_mode": "eager",
                "enforce_eager": True,
                "repetition": repetition,
                "latency_seconds": 2.0,
                "output_throughput_tokens_per_sec": throughput,
                "request_throughput_per_sec": 5.0,
                "avg_time_per_sample": 0.2,
                "acceptance_rate": 0.6,
                "mean_acceptance_length": 2.0,
                "num_drafts": 10,
                "num_draft_tokens": 30,
                "num_accepted_tokens": 18,
                "timing": {
                    "overall": {"wall_seconds": 2.0},
                    "draft": {"draft_seconds": 0.4},
                    "target": {"target_seconds": 0.8},
                    "runtime": {},
                },
            }
        )

    summary = BENCHMARK.calculate_repetition_summary(records, _args())

    assert summary["repetitions"] == 3
    assert (
        summary["statistics"]["output_throughput_tokens_per_sec"]["mean"] == 110
    )
    assert summary["timing_statistics"]["draft"]["draft_seconds"]["mean"] == 0.4
    assert summary["timing_statistics"]["target"]["target_seconds"]["mean"] == 0.8
