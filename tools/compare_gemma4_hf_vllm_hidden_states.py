#!/usr/bin/env python3
"""Compare Gemma-4 auxiliary hidden states from HF and vLLM."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

_torchvision_schema_library = None
if os.environ.get("ANGELSLIM_SKIP_TORCHVISION_SCHEMA") != "1":
    try:
        getattr(torch.ops.torchvision, "nms")
    except AttributeError:
        _torchvision_schema_library = torch.library.Library(
            "torchvision", "FRAGMENT"
        )
        _torchvision_schema_library.define(
            "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
        )
        _torchvision_schema_library.define(
            "qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
        )

from transformers import AutoProcessor

from angelslim.compressor.speculative import create_target_model


def layer_metrics(reference, candidate, layer_ids):
    hidden_size = reference.shape[-1] // len(layer_ids)
    metrics = []
    for slot, layer_id in enumerate(layer_ids):
        left = reference[..., slot * hidden_size : (slot + 1) * hidden_size].float()
        right = candidate[..., slot * hidden_size : (slot + 1) * hidden_size].float()
        difference = right - left
        metrics.append(
            {
                "decoder_layer": layer_id,
                "cosine": float(F.cosine_similarity(left.flatten(), right.flatten(), dim=0)),
                "mean_abs_error": float(difference.abs().mean()),
                "max_abs_error": float(difference.abs().max()),
                "hf_rms": float(left.square().mean().sqrt()),
                "vllm_rms": float(right.square().mean().sqrt()),
            }
        )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Explain in one sentence why speculative decoding can improve throughput.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--draft-config",
        type=Path,
        default=None,
        help="Read target layer IDs from a generated DFlare config.",
    )
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--vllm-only",
        action="store_true",
        help="Validate vLLM hidden shape/finite values when packed AWQ cannot run in HF.",
    )
    args = parser.parse_args()
    expected_width = None
    if args.layers is None:
        if args.draft_config is None:
            raise ValueError("Pass --layers or --draft-config")
        draft_config = json.loads(args.draft_config.read_text(encoding="utf-8"))
        args.layers = list(draft_config["dflare_config"]["target_layer_ids"])
        expected_width = (
            len(args.layers) * int(draft_config["target_hidden_size"])
        )

    processor = AutoProcessor.from_pretrained(args.model)
    token_ids = processor.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    hf_hidden = None
    if not args.vllm_only:
        hf_target = create_target_model(
            backend="hf",
            model_path=args.model,
            modal_type="Gemma4",
            target_model_type="gemma4",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        hf_hidden = hf_target.get_hidden_states_only(
            input_ids=input_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
            aux_hidden_states_layer_ids=args.layers,
        ).cpu()
        del hf_target
        gc.collect()
        torch.cuda.empty_cache()

    vllm_target = create_target_model(
        backend="vllm",
        model_path=args.model,
        modal_type="VLM",
        target_model_type="gemma4",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    vllm_hidden = vllm_target.get_aux_and_target_hiddens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aux_hidden_states_layer_ids=args.layers,
    )["hidden_states"].cpu()
    if expected_width is not None and vllm_hidden.shape[-1] != expected_width:
        raise ValueError(
            f"vLLM hidden width {vllm_hidden.shape[-1]} != expected {expected_width}"
        )
    result: dict = {
        "model": args.model,
        "prompt": args.prompt,
        "input_tokens": len(token_ids),
        "shape": list(vllm_hidden.shape),
        "decoder_layers": args.layers,
        "vllm_only": args.vllm_only,
        "all_finite": bool(torch.isfinite(vllm_hidden).all()),
    }
    if hf_hidden is not None:
        old_shifted_layers = [layer_id - 1 for layer_id in args.layers]
        shifted_hidden = vllm_target.get_aux_and_target_hiddens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aux_hidden_states_layer_ids=old_shifted_layers,
        )["hidden_states"].cpu()
        result["correct_mapping"] = layer_metrics(
            hf_hidden, vllm_hidden, args.layers
        )
        result["old_shifted_mapping"] = layer_metrics(
            hf_hidden,
            shifted_hidden,
            old_shifted_layers,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
