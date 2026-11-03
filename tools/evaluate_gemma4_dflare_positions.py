#!/usr/bin/env python3
"""Read-only per-position evaluation for Gemma 4 DFlare offline caches.

The evaluator loads a final Hugging Face draft checkpoint and the target
model's frozen embedding/lm-head components. It never constructs a Trainer,
optimizer, or checkpoint writer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer

from angelslim.compressor.speculative.train.models.draft import (
    Gemma4DFlareDraftModel,
)
from angelslim.compressor.speculative.train.trainer.dflash_position_metrics import (
    add_dflash_position_counts,
    dflash_position_metric_counts,
    empty_dflash_position_counts,
    finalize_dflash_position_metrics,
    select_deterministic_anchors,
)
from angelslim.compressor.speculative.train.trainer.online_dflash_trainer import (
    TargetEmbeddingsAndHead,
    create_dflash_block_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gemma 4 DFlare per-position accuracy and CE from "
            "precomputed offline caches."
        )
    )
    parser.add_argument("--draft-checkpoint", required=True)
    parser.add_argument("--target-model-name-or-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-pattern", default="*.ckpt")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--num-anchors",
        type=int,
        help="Maximum deterministic anchors per sample (default: draft config).",
    )
    parser.add_argument("--anchor-seed", type=int, default=0)
    parser.add_argument(
        "--anchor-batch-size",
        type=int,
        default=16,
        help="Number of anchor blocks per forward pass.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, for example cuda, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("eager", "sdpa", "flex_attention"),
        default="eager",
        help="Dense SDPA is used for eager/sdpa; flex uses the trainer BlockMask.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--audit-loss-mask-spans",
        action="store_true",
        help="Decode contiguous true loss-mask spans into the JSON report.",
    )
    parser.add_argument("--audit-max-samples", type=int, default=3)
    parser.add_argument("--audit-max-spans-per-sample", type=int, default=20)
    parser.add_argument(
        "--record-predictions",
        action="store_true",
        help="Include a bounded set of anchor-level target/predicted token IDs.",
    )
    parser.add_argument("--max-recorded-anchors", type=int, default=16)
    parser.add_argument(
        "--intra-block-causal-diagnostic",
        action="store_true",
        help="Diagnostic only: restrict each draft query to earlier positions in its block.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional report path. Checkpoints and cache files remain untouched.",
    )
    return parser.parse_args()


def _dense_dflash_attention_mask(
    anchors: torch.Tensor,
    context_length: int,
    block_size: int,
    device: torch.device,
    intra_block_causal: bool = False,
) -> torch.Tensor:
    """Build the trainer's context-causal/intra-block mask as a dense bool mask."""
    draft_length = anchors.numel() * block_size
    query_indices = torch.arange(draft_length, device=device)
    query_blocks = query_indices // block_size
    key_indices = torch.arange(context_length + draft_length, device=device)

    context_allowed = (
        key_indices.view(1, -1) < anchors[query_blocks].view(-1, 1)
    ) & (key_indices.view(1, -1) < context_length)
    draft_key_blocks = (key_indices - context_length).clamp_min(0) // block_size
    draft_allowed = (key_indices.view(1, -1) >= context_length) & (
        draft_key_blocks.view(1, -1) == query_blocks.view(-1, 1)
    )
    if intra_block_causal:
        query_offsets = query_indices % block_size
        key_offsets = (key_indices - context_length).clamp_min(0) % block_size
        draft_allowed = draft_allowed & (
            key_offsets.view(1, -1) <= query_offsets.view(-1, 1)
        )
    return (context_allowed | draft_allowed).unsqueeze(0).unsqueeze(0)


def _attention_mask(
    anchors: torch.Tensor,
    context_length: int,
    block_size: int,
    backend: str,
    device: torch.device,
    intra_block_causal: bool = False,
):
    if backend == "flex_attention":
        if intra_block_causal:
            raise ValueError(
                "--intra-block-causal-diagnostic currently requires eager or sdpa"
            )
        return create_dflash_block_mask(
            anchor_positions=anchors.unsqueeze(0),
            block_keep_mask=torch.ones(
                (1, anchors.numel()),
                dtype=torch.bool,
                device=device,
            ),
            S=context_length,
            block_size=block_size,
            device=device,
        )
    return _dense_dflash_attention_mask(
        anchors=anchors,
        context_length=context_length,
        block_size=block_size,
        device=device,
        intra_block_causal=intra_block_causal,
    )


def _load_cache(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    required = ("input_ids", "hidden_states", "loss_mask")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required cache keys: {missing}")
    if any(not isinstance(data[key], torch.Tensor) for key in required):
        raise TypeError(f"{path} cache values for {required} must be tensors")

    input_ids = data["input_ids"]
    hidden_states = data["hidden_states"]
    loss_mask = data["loss_mask"]
    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids.squeeze(0)
    if hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
        hidden_states = hidden_states.squeeze(0)
    if loss_mask.ndim == 2 and loss_mask.shape[0] == 1:
        loss_mask = loss_mask.squeeze(0)
    if input_ids.ndim != 1 or loss_mask.ndim != 1 or hidden_states.ndim != 2:
        raise ValueError(
            f"{path} has unsupported shapes: input_ids={tuple(input_ids.shape)}, "
            f"hidden_states={tuple(hidden_states.shape)}, "
            f"loss_mask={tuple(loss_mask.shape)}"
        )
    if not (input_ids.shape[0] == hidden_states.shape[0] == loss_mask.shape[0]):
        raise ValueError(f"{path} cache sequence lengths do not match")
    return {
        "input_ids": input_ids.long(),
        "hidden_states": hidden_states,
        "loss_mask": loss_mask,
    }


def _loss_mask_spans(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    tokenizer: Any,
    max_spans: int,
) -> list[dict[str, Any]]:
    mask = loss_mask > 0.5
    padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.bool),
            mask.cpu(),
            torch.zeros(1, dtype=torch.bool),
        )
    )
    transitions = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
    starts = torch.nonzero(transitions == 1, as_tuple=False).flatten()
    ends = torch.nonzero(transitions == -1, as_tuple=False).flatten()
    spans = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        token_ids = input_ids[start:end].tolist()
        spans.append(
            {
                "start": start,
                "end_exclusive": end,
                "token_count": end - start,
                "decoded_text": tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                ),
            }
        )
        if len(spans) >= max_spans:
            break
    return spans


def _load_tokenizer(model_path: str, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
    except (OSError, ValueError):
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        return processor.tokenizer


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.anchor_batch_size < 1:
        raise ValueError("--anchor-batch-size must be positive")
    if args.max_recorded_anchors < 1:
        raise ValueError("--max-recorded-anchors must be positive")

    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]

    cache_files = sorted(Path(args.cache_dir).rglob(args.cache_pattern))
    if not cache_files:
        raise ValueError(
            f"No {args.cache_pattern!r} files found under {args.cache_dir}"
        )
    if args.max_samples is not None:
        cache_files = cache_files[: args.max_samples]

    draft_model = Gemma4DFlareDraftModel.from_pretrained(
        args.draft_checkpoint,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    draft_model.config._attn_implementation = args.attention_backend
    draft_model.eval()

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_name_or_path,
        device=str(device),
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    target_embed_tokens = target_components.embed_tokens
    target_lm_head = target_components.lm_head

    block_size = int(draft_model.block_size)
    num_anchors = (
        args.num_anchors
        if args.num_anchors is not None
        else int(getattr(draft_model.config, "num_anchors", 512))
    )
    if num_anchors < 1:
        raise ValueError("--num-anchors must be positive")

    counts = empty_dflash_position_counts(block_size, device=device)
    evaluated_samples = 0
    evaluated_anchors = 0
    samples_without_anchors = 0
    audits = []
    prediction_records = []
    captured_layer_outputs = []
    layer_hook_handles = []
    captured_components = {}
    component_hook_handles = []
    if args.record_predictions:
        for layer in draft_model.layers:
            layer_hook_handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output: captured_layer_outputs.append(
                        output.detach()
                    )
                )
            )
        first_layer = draft_model.layers[0]
        for name, module in (
            ("input_normed", first_layer.input_layernorm),
            ("attention_output", first_layer.attention),
            ("post_normed", first_layer.post_attention_layernorm),
            ("mlp_output", first_layer.down_proj),
        ):
            captured_components[name] = []
            component_hook_handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, component=name: (
                        captured_components[component].append(output.detach())
                    )
                )
            )
    tokenizer: Optional[Any] = None
    if args.audit_loss_mask_spans:
        tokenizer = _load_tokenizer(
            args.target_model_name_or_path,
            args.trust_remote_code,
        )

    for sample_index, cache_path in enumerate(cache_files):
        cache = _load_cache(cache_path)
        input_ids_cpu = cache["input_ids"]
        loss_mask_cpu = cache["loss_mask"]
        anchors_cpu = select_deterministic_anchors(
            loss_mask=loss_mask_cpu,
            block_size=block_size,
            num_anchors=num_anchors,
            seed=args.anchor_seed,
            sample_index=sample_index,
        )

        if tokenizer is not None and sample_index < args.audit_max_samples:
            audits.append(
                {
                    "cache_file": str(cache_path),
                    "spans": _loss_mask_spans(
                        input_ids_cpu,
                        loss_mask_cpu,
                        tokenizer,
                        args.audit_max_spans_per_sample,
                    ),
                }
            )

        if anchors_cpu.numel() == 0:
            samples_without_anchors += 1
            continue

        input_ids = input_ids_cpu.to(device)
        hidden_states = cache["hidden_states"].unsqueeze(0).to(device=device, dtype=dtype)
        loss_mask = loss_mask_cpu.to(device)
        sequence_length = input_ids.numel()
        context_position_ids = torch.arange(
            sequence_length,
            device=device,
        ).unsqueeze(0)

        for start in range(0, anchors_cpu.numel(), args.anchor_batch_size):
            anchors = anchors_cpu[start : start + args.anchor_batch_size].to(device)
            anchor_count = anchors.numel()
            offsets = torch.arange(block_size, device=device).view(1, -1)
            label_indices = anchors.view(-1, 1) + offsets
            target_ids = input_ids[label_indices]
            valid_mask = loss_mask[label_indices]
            valid_mask[:, 0] = 0

            noise_ids = torch.full(
                (anchor_count, block_size),
                int(draft_model.mask_token_id),
                dtype=torch.long,
                device=device,
            )
            noise_ids[:, 0] = input_ids[anchors]
            noise_embedding = target_embed_tokens(
                noise_ids.reshape(1, -1)
            ) * target_components.embedding_scale
            draft_position_ids = label_indices.reshape(1, -1)
            position_ids = torch.cat(
                (context_position_ids, draft_position_ids),
                dim=1,
            )
            attention_mask = _attention_mask(
                anchors=anchors,
                context_length=sequence_length,
                block_size=block_size,
                backend=args.attention_backend,
                device=device,
                intra_block_causal=args.intra_block_causal_diagnostic,
            )

            captured_layer_outputs.clear()
            for values in captured_components.values():
                values.clear()
            output_hidden = draft_model(
                noise_embedding=noise_embedding.to(dtype),
                target_hidden=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            logits = target_lm_head(
                output_hidden.to(target_lm_head.weight.dtype)
            )
            if target_components.logit_softcap is not None:
                logits = torch.tanh(
                    logits / target_components.logit_softcap
                ) * target_components.logit_softcap

            flat_targets = target_ids.reshape(-1)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            token_losses = F.cross_entropy(
                flat_logits,
                flat_targets,
                reduction="none",
            ).view(anchor_count, block_size)
            predictions = flat_logits.argmax(dim=-1).view(anchor_count, block_size)
            if args.record_predictions and len(prediction_records) < args.max_recorded_anchors:
                remaining = args.max_recorded_anchors - len(prediction_records)
                for row in range(min(anchor_count, remaining)):
                    hidden_row = output_hidden.view(
                        anchor_count,
                        block_size,
                        output_hidden.shape[-1],
                    )[row]
                    prediction_records.append(
                        {
                            "cache_file": str(cache_path),
                            "anchor": int(anchors[row].item()),
                            "noise_ids": noise_ids[row].detach().cpu().tolist(),
                            "embed_row_sums": noise_embedding.view(
                                anchor_count,
                                block_size,
                                noise_embedding.shape[-1],
                            )[row].float().sum(-1).detach().cpu().tolist(),
                            "embed_row_norms": noise_embedding.view(
                                anchor_count,
                                block_size,
                                noise_embedding.shape[-1],
                            )[row].float().norm(dim=-1).detach().cpu().tolist(),
                            "output_row_sums": hidden_row.float().sum(-1).detach().cpu().tolist(),
                            "output_row_norms": hidden_row.float().norm(dim=-1).detach().cpu().tolist(),
                            "layers": [
                                {
                                    "row_sums": layer_output.view(
                                        anchor_count,
                                        block_size,
                                        layer_output.shape[-1],
                                    )[row].float().sum(-1).detach().cpu().tolist(),
                                    "row_norms": layer_output.view(
                                        anchor_count,
                                        block_size,
                                        layer_output.shape[-1],
                                    )[row].float().norm(dim=-1).detach().cpu().tolist(),
                                }
                                for layer_output in captured_layer_outputs
                            ],
                            "layer0_components": {
                                component: {
                                    "row_sums": values[-1].view(
                                        anchor_count,
                                        block_size,
                                        values[-1].shape[-1],
                                    )[row].float().sum(-1).detach().cpu().tolist(),
                                    "row_norms": values[-1].view(
                                        anchor_count,
                                        block_size,
                                        values[-1].shape[-1],
                                    )[row].float().norm(dim=-1).detach().cpu().tolist(),
                                }
                                for component, values in captured_components.items()
                            },
                            "target_ids": target_ids[row].detach().cpu().tolist(),
                            "predicted_ids": predictions[row].detach().cpu().tolist(),
                        }
                    )
            add_dflash_position_counts(
                counts,
                dflash_position_metric_counts(
                    predictions=predictions,
                    targets=target_ids,
                    valid_mask=valid_mask,
                    token_losses=token_losses,
                ),
            )
            evaluated_anchors += anchor_count

        evaluated_samples += 1

    for handle in layer_hook_handles:
        handle.remove()
    for handle in component_hook_handles:
        handle.remove()

    report = {
        "draft_checkpoint": str(Path(args.draft_checkpoint)),
        "target_model_name_or_path": args.target_model_name_or_path,
        "cache_dir": str(Path(args.cache_dir)),
        "block_size": block_size,
        "anchor_seed": args.anchor_seed,
        "num_anchors_per_sample_limit": num_anchors,
        "sample_limit": args.max_samples,
        "selected_cache_files": len(cache_files),
        "evaluated_samples": evaluated_samples,
        "samples_without_anchors": samples_without_anchors,
        "evaluated_anchors": evaluated_anchors,
        "positions": finalize_dflash_position_metrics(counts),
    }
    if args.audit_loss_mask_spans:
        report["loss_mask_span_audit"] = audits
    if args.record_predictions:
        report["anchor_predictions"] = prediction_records
    return report


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
