#!/usr/bin/env python3
"""Mine Gemma DFlare hard-tail training rows from unique RoboVQA conversations.

A serving-like greedy walk over checkpoint-120000 drafts records every block
that accepts 8-14 of the 15 speculative tokens. Each event becomes one JSONL
row with the original conversation plus ``hard_tail_assistant_token_index`` so
online training can restrict ``loss_mask`` to that 16-token window.

Usage:
  torchrun --standalone --nproc_per_node=6 tools/mine_dflare_hard_tails.py \\
      --target_model_name_or_path /path/to/gemma-4-E4B-it-AWQ-INT4 \\
      --draft_model_name_or_path /path/to/checkpoint-120000 \\
      --train_data_path /path/to/unique/angelslim_train.jsonl \\
      --output_dir /path/to/hard-tail-jsonl-dir
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from safetensors.torch import load_file

# Some ROCm images expose torchvision transforms without detection ops.
if os.environ.get("ANGELSLIM_SKIP_TORCHVISION_SCHEMA") != "1":
    try:
        getattr(torch.ops.torchvision, "nms")
    except AttributeError:
        _torchvision_schema_library = torch.library.Library("torchvision", "FRAGMENT")
        _torchvision_schema_library.define(
            "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
        )
        _torchvision_schema_library.define(
            "qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
        )

from angelslim.compressor.speculative import (
    DraftModelConfig,
    create_draft_model,
    create_target_model,
)
from angelslim.compressor.speculative.train.data.chat_templates import (
    ChatTemplateType,
    string_to_chat_template_type,
)
from angelslim.compressor.speculative.train.data.dataset_builder.online_dataset_builder import (
    OnlineGemma4VLMDatasetBuilder,
    OnlineVLMDatasetBuilder,
)
from angelslim.compressor.speculative.train.trainer.online_dflash_trainer import (
    TargetEmbeddingsAndHead,
    create_dflash_block_mask,
)
from angelslim.utils import rank0_print


TARGET_FORWARD_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "input_features",
    "input_features_mask",
    "input_position_ids",
    "image_position_ids",
    "raw_images",
    "mm_token_type_ids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine DFlare B16 hard-tail rows")
    parser.add_argument("--target_model_name_or_path", required=True)
    parser.add_argument("--draft_model_name_or_path", required=True)
    parser.add_argument(
        "--draft_model_config_path",
        default=None,
        help="Defaults to <draft_model_name_or_path>/config.json",
    )
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--chat_template_type", default="gemma4")
    parser.add_argument("--model_max_length", type=int, default=3072)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--min_k", type=int, default=8)
    parser.add_argument("--max_k", type=int, default=14)
    parser.add_argument(
        "--serve_k",
        type=int,
        default=None,
        help="Speculative tokens per serving step for the walk (default block_size-1)",
    )
    parser.add_argument("--attention_backend", default="flex_attention")
    parser.add_argument("--max_anchors_per_forward", type=int, default=256)
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    return parser.parse_args()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_global_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def init_distributed() -> None:
    if get_world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(get_local_rank())


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


class DraftBlockScorer:
    """Greedy DFlare block scorer matching OnlineDFlashTrainer inference."""

    def __init__(
        self,
        draft_model: torch.nn.Module,
        embeddings: TargetEmbeddingsAndHead,
        block_size: int,
        mask_token_id: int,
        attention_backend: str,
    ):
        self.model = draft_model
        self.target_embed_tokens = embeddings.embed_tokens
        self.target_embedding_scale = embeddings.embedding_scale
        self.target_lm_head = embeddings.lm_head
        self.target_logit_softcap = embeddings.logit_softcap
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self.attention_backend = attention_backend
        self.max_anchors_per_forward = 256

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        return (anchor_positions.unsqueeze(-1) + offsets).view(
            anchor_positions.shape[0], -1
        )

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        n_blocks = anchor_positions.shape[1]
        device = input_ids.device
        noise_ids = torch.full(
            (bsz, n_blocks * self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        block_starts = (
            torch.arange(n_blocks, device=device) * self.block_size
        ).unsqueeze(0).expand(bsz, -1)
        anchor_tokens = torch.gather(
            input_ids, 1, anchor_positions.clamp(0, seq_len - 1)
        )
        batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n_blocks)
        noise_ids[batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.full_like(anchor_tokens, self.mask_token_id),
        )
        return self.target_embed_tokens(noise_ids) * self.target_embedding_scale

    @torch.no_grad()
    def accepted_prefix_lengths(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        anchors: List[int],
        token_position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[int, int]:
        if not anchors:
            return {}
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        accepted: Dict[int, int] = {}
        model_dtype = next(self.model.parameters()).dtype
        chunk_size = max(int(self.max_anchors_per_forward), 1)
        for start in range(0, len(anchors), chunk_size):
            chunk = anchors[start : start + chunk_size]
            n_blocks = len(chunk)
            anchor_positions = torch.tensor(
                [chunk], dtype=torch.long, device=device
            )
            block_keep_mask = torch.ones(
                (bsz, n_blocks), dtype=torch.bool, device=device
            )
            noise_embedding = self._create_noise_embed(
                input_ids, anchor_positions, block_keep_mask
            )
            if token_position_ids is not None:
                # Qwen M-RoPE (3, B, S): the block continues from the anchor's
                # own time/height/width position, as in training and serve.
                context_position_ids = token_position_ids
                anchor_mrope = context_position_ids[:, :, anchor_positions[0]]
                offsets = torch.arange(self.block_size, device=device)
                draft_position_ids = (
                    anchor_mrope.unsqueeze(-1) + offsets
                ).reshape(3, bsz, -1)
                full_position_ids = torch.cat(
                    [context_position_ids, draft_position_ids], dim=-1
                )
            else:
                context_position_ids = torch.arange(
                    seq_len, device=device
                ).unsqueeze(0).expand(bsz, -1)
                draft_position_ids = self._create_position_ids(anchor_positions)
                full_position_ids = torch.cat(
                    [context_position_ids, draft_position_ids], dim=1
                )
            attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
                causal_within_block=False,
                attention_backend=self.attention_backend,
            )
            # Training runs the draft under bf16 autocast; Qwen M-RoPE cos/sin
            # are float32 and only mix with bf16 activations inside autocast.
            with torch.autocast("cuda", dtype=model_dtype):
                output_hidden = self.model(
                    noise_embedding=noise_embedding.to(model_dtype),
                    target_hidden=hidden_states.to(model_dtype),
                    attention_mask=attn_mask,
                    position_ids=full_position_ids,
                )
            logits = self.target_lm_head(
                output_hidden.to(self.target_lm_head.weight.dtype)
            )
            if self.target_logit_softcap is not None:
                logits = logits / self.target_logit_softcap
                logits = torch.tanh(logits) * self.target_logit_softcap
            logits = logits.view(bsz, n_blocks, self.block_size, -1)
            predictions = logits.argmax(dim=-1)[0]
            offsets = torch.arange(self.block_size, device=device).view(1, -1)
            label_indices = (anchor_positions[0].unsqueeze(-1) + offsets).clamp(
                max=seq_len - 1
            )
            targets = input_ids[0].index_select(0, label_indices.reshape(-1)).view(
                n_blocks, self.block_size
            )
            matches = predictions == targets
            tail = matches[:, 1:]
            failed = ~tail
            any_fail = failed.any(dim=-1)
            first_fail = failed.to(torch.int64).argmax(dim=-1)
            ks = torch.where(
                any_fail,
                first_fail,
                torch.full_like(first_fail, self.block_size - 1),
            )
            for anchor, k in zip(chunk, ks.tolist()):
                accepted[int(anchor)] = int(k)
        return accepted


def serving_walk_hard_tails(
    loss_mask: torch.Tensor,
    k_by_anchor: Dict[int, int],
    seq_len: int,
    block_size: int,
    min_k: int,
    max_k: int,
    serve_k: Optional[int] = None,
    histogram: Optional[Counter] = None,
) -> List[Tuple[int, int, int]]:
    """Return (anchor_pos, assistant_index, accepted_k) for min_k..max_k accepts.

    The walk commits what serving with ``serve_k`` draft tokens would: the
    accepted prefix (capped at ``serve_k``) plus one target token.
    """
    serve_k = block_size - 1 if serve_k is None else int(serve_k)
    assistant = (loss_mask.view(-1) > 0.5).nonzero(as_tuple=False).view(-1)
    if assistant.numel() == 0:
        return []
    assistant_index = {
        int(position): idx for idx, position in enumerate(assistant.tolist())
    }
    pos = max(int(assistant[0].item()) - 1, 0)
    events: List[Tuple[int, int, int]] = []
    seen = set()
    for _ in range(seq_len):
        if pos in seen or pos + block_size > seq_len:
            break
        seen.add(pos)
        if pos not in k_by_anchor:
            break
        accepted_k = min(int(k_by_anchor[pos]), serve_k)
        if histogram is not None:
            histogram[accepted_k] += 1
        if min_k <= accepted_k <= max_k:
            ht_index = assistant_index.get(pos, -1)
            events.append((pos, ht_index, accepted_k))
        pos += accepted_k + 1
    return events


def iter_rank_records(
    path: str,
    rank: int,
    world_size: int,
    sample_num: Optional[int],
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        for source_idx, line in enumerate(handle):
            if sample_num is not None and source_idx >= sample_num:
                break
            if source_idx % world_size != rank:
                continue
            line = line.strip()
            if not line:
                continue
            yield source_idx, json.loads(line)


def load_progress(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(int(line))
    return done


def merge_shards(output_dir: Path, world_size: int) -> Dict[str, Any]:
    merged_path = output_dir / "angelslim_train.jsonl"
    histogram: Counter[int] = Counter()
    seen_ids = set()
    n_events = 0
    with open(merged_path, "w", encoding="utf-8") as dest:
        for rank in range(world_size):
            shard = output_dir / "shards" / f"rank{rank}.jsonl"
            if not shard.is_file():
                continue
            with open(shard, "r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    record_id = record.get("id")
                    if record_id in seen_ids:
                        continue
                    seen_ids.add(record_id)
                    dest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    histogram[int(record.get("accepted_k", -1))] += 1
                    n_events += 1
    stats = {
        "events": n_events,
        "unique_ids": len(seen_ids),
        "k_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "merged_path": str(merged_path),
    }
    if n_events:
        stats["mean_k"] = sum(k * n for k, n in histogram.items()) / n_events
    (output_dir / "mine_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    args = parse_args()
    init_distributed()
    rank = get_global_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    if rank == 0:
        shard_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    draft_config_path = args.draft_model_config_path or str(
        Path(args.draft_model_name_or_path) / "config.json"
    )
    draft_model_config = DraftModelConfig.from_file(draft_config_path)
    dflare_config = (
        getattr(draft_model_config, "dflare_config", None)
        or getattr(draft_model_config, "dflash_config", None)
        or {}
    )
    target_layer_ids = dflare_config.get("target_layer_ids")
    if not target_layer_ids:
        raise ValueError("dflare_config.target_layer_ids is required")
    mask_token_id = int(
        dflare_config.get(
            "mask_token_id",
            getattr(draft_model_config, "mask_token_id", 4),
        )
    )
    draft_model_config.attention_backend = args.attention_backend
    draft_model_config._attn_implementation = args.attention_backend
    draft_model_config.aux_hidden_states_layer_ids = target_layer_ids

    target_model_type = getattr(draft_model_config, "target_model_type", None) or "gemma4"
    is_gemma = target_model_type == "gemma4"
    rank0_print(f"Loading target model ({target_model_type})...")
    target_model = create_target_model(
        backend=args.target_backend,
        model_path=args.target_model_name_or_path,
        modal_type="Gemma4" if is_gemma else "VLM",
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        target_model_type=target_model_type,
    )
    rank0_print("Loading draft model...")
    draft_model = create_draft_model(draft_model_config)
    weights_path = Path(args.draft_model_name_or_path) / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    load_result = draft_model.load_state_dict(
        load_file(str(weights_path), device="cpu"),
        strict=True,
    )
    rank0_print(
        f"Loaded draft weights missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)}"
    )
    draft_model.to(device=device, dtype=torch_dtype)
    draft_model.eval()
    draft_model.requires_grad_(False)

    embeddings = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_name_or_path,
        embed_key=getattr(
            draft_model_config,
            "embed_weight_key",
            "model.language_model.embed_tokens.weight",
        ),
        device=str(device),
        dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    scorer = DraftBlockScorer(
        draft_model=draft_model,
        embeddings=embeddings,
        block_size=args.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
    )
    scorer.max_anchors_per_forward = args.max_anchors_per_forward
    if is_gemma:
        builder = OnlineGemma4VLMDatasetBuilder(
            tokenizer=target_model.tokenizer,
            max_length=args.model_max_length,
            chat_template_type=ChatTemplateType.GEMMA4,
        )
    else:
        builder = OnlineVLMDatasetBuilder(
            tokenizer=target_model.tokenizer,
            max_length=args.model_max_length,
            chat_template_type=string_to_chat_template_type(args.chat_template_type),
        )
    collator = builder.get_data_collator()
    # Same lookup as OnlineDFlashTrainer._serving_position_ids.
    rope_backend = getattr(target_model, "backend", target_model)
    rope_model = getattr(getattr(rope_backend, "model", None), "model", None)
    if not hasattr(rope_model, "get_rope_index"):
        rope_model = None
    walk_histogram: Counter[int] = Counter()

    shard_path = shard_dir / f"rank{rank}.jsonl"
    progress_path = shard_dir / f"rank{rank}.progress"
    done = load_progress(progress_path) if args.resume else set()
    shard_mode = "a" if args.resume and shard_path.is_file() else "w"
    n_sources = 0
    n_events = 0
    n_errors = 0
    histogram: Counter[int] = Counter()
    t0 = time.time()

    rank0_print(
        f"Mining hard tails from {args.train_data_path} "
        f"(rank {rank}/{world_size}, resume={args.resume}, done={len(done)})"
    )
    with open(shard_path, shard_mode, encoding="utf-8") as shard_handle, open(
        progress_path, "a", encoding="utf-8"
    ) as progress_handle:
        for source_idx, record in iter_rank_records(
            args.train_data_path, rank, world_size, args.sample_num
        ):
            if source_idx in done:
                continue
            n_sources += 1
            try:
                processed = builder._process_single_conversation(
                    record["conversations"]
                )
                if processed is None:
                    progress_handle.write(f"{source_idx}\n")
                    progress_handle.flush()
                    continue
                batch = move_batch_to_device(collator([processed]), device)
                input_ids = batch["input_ids"]
                if input_ids.dim() == 3:
                    input_ids = input_ids.squeeze(1)
                attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.squeeze(1)
                loss_mask = batch["loss_mask"]
                if loss_mask.dim() == 3:
                    loss_mask = loss_mask.squeeze(1)
                seq_len = int(input_ids.shape[-1])
                if seq_len < args.block_size + 1:
                    progress_handle.write(f"{source_idx}\n")
                    progress_handle.flush()
                    continue
                target_kwargs = {
                    key: batch[key]
                    for key in TARGET_FORWARD_KEYS
                    if key in batch and batch[key] is not None
                }
                hidden_states = target_model.get_hidden_states_only(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    aux_hidden_states_layer_ids=target_layer_ids,
                    **target_kwargs,
                )
                if hidden_states.shape[1] != seq_len:
                    raise RuntimeError(
                        f"hidden_states length {hidden_states.shape[1]} "
                        f"!= input_ids length {seq_len}"
                    )
                assistant = (loss_mask.view(-1) > 0.5).nonzero(as_tuple=False).view(-1)
                if assistant.numel() == 0:
                    progress_handle.write(f"{source_idx}\n")
                    progress_handle.flush()
                    continue
                token_position_ids = None
                if rope_model is not None and batch.get("mm_token_type_ids") is not None:
                    token_position_ids, _ = rope_model.get_rope_index(
                        input_ids,
                        batch["mm_token_type_ids"],
                        image_grid_thw=batch.get("image_grid_thw"),
                        video_grid_thw=batch.get("video_grid_thw"),
                        attention_mask=attention_mask,
                    )
                    token_position_ids = token_position_ids.to(
                        device=device, dtype=torch.long
                    )
                start = max(int(assistant[0].item()) - 1, 0)
                anchors = list(range(start, seq_len - args.block_size + 1))
                k_by_anchor = scorer.accepted_prefix_lengths(
                    input_ids, hidden_states, anchors, token_position_ids
                )
                events = serving_walk_hard_tails(
                    loss_mask=loss_mask,
                    k_by_anchor=k_by_anchor,
                    seq_len=seq_len,
                    block_size=args.block_size,
                    min_k=args.min_k,
                    max_k=args.max_k,
                    serve_k=args.serve_k,
                    histogram=walk_histogram,
                )
                source_id = str(record.get("id", source_idx))
                for _anchor_pos, ht_index, accepted_k in events:
                    event = {
                        "id": f"{source_id}::ht{ht_index}::k{accepted_k}",
                        "conversations": record["conversations"],
                        "hard_tail_assistant_token_index": ht_index,
                        "accepted_k": accepted_k,
                        "source_id": source_id,
                    }
                    shard_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                    histogram[accepted_k] += 1
                    n_events += 1
                shard_handle.flush()
            except Exception as exc:
                n_errors += 1
                rank0_print(f"[rank {rank}] source {source_idx} failed: {exc}")
                if n_errors == 1:
                    import traceback

                    rank0_print(traceback.format_exc())
            progress_handle.write(f"{source_idx}\n")
            progress_handle.flush()
            if n_sources % 20 == 0:
                elapsed = time.time() - t0
                rate = n_sources / elapsed if elapsed else 0.0
                rank0_print(
                    f"[rank {rank}] sources={n_sources} events={n_events} "
                    f"errors={n_errors} {rate:.2f} src/s histogram={dict(histogram)} "
                    f"walk={dict(sorted(walk_histogram.items()))}"
                )

    elapsed = time.time() - t0
    rank_stats = {
        "rank": rank,
        "sources": n_sources,
        "events": n_events,
        "errors": n_errors,
        "elapsed_s": elapsed,
        "k_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "walk_histogram": {str(k): walk_histogram[k] for k in sorted(walk_histogram)},
    }
    (shard_dir / f"rank{rank}.stats.json").write_text(
        json.dumps(rank_stats, indent=2) + "\n", encoding="utf-8"
    )
    rank0_print(f"[rank {rank}] finished {rank_stats}")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        stats = merge_shards(output_dir, world_size)
        rank0_print(f"Merged hard-tail corpus: {stats}")


if __name__ == "__main__":
    main()
