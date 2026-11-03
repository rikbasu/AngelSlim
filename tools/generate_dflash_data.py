#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# DFlash offline data pre-generation script.
#
# Usage:
#   torchrun --nproc_per_node=8 tools/generate_dflash_data.py \
#       --target_model_name_or_path /path/to/Qwen3-4B \
#       --draft_model_config_path configs/qwen3_dflash.json \
#       --train_data_path /path/to/data.jsonl \
#       --output_dir /path/to/output/ckpts \
#       --model_max_length 3072 \
#       --chat_template_type qwen3
#
# Each output .ckpt file contains:
#   - input_ids:      LongTensor  [1, S]
#   - hidden_states:  BFloat16Tensor [1, S, D*num_target_layers]  (multi-layer concat)
#   - loss_mask:      LongTensor  [1, S]
#   - attention_mask: LongTensor  [1, S]

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    create_target_model,
    get_supported_chat_template_type_strings,
)
from angelslim.utils import rank0_print


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-generate DFlash training data (hidden states) from target model"
    )

    # Model
    parser.add_argument("--target_model_name_or_path", type=str, required=True)
    parser.add_argument("--draft_model_config_path", type=str, required=True)
    parser.add_argument(
        "--target_backend",
        type=str,
        default="hf",
        choices=["hf", "vllm"],
        help="Target model backend",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    # Data
    parser.add_argument(
        "--train_data_path", type=str, nargs="+", required=True, help="Input JSONL file(s)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save .ckpt files"
    )
    parser.add_argument(
        "--chat_template_type",
        type=str,
        default="qwen3",
        help=f"Supported: {', '.join(get_supported_chat_template_type_strings())}",
    )
    parser.add_argument("--model_max_length", type=int, default=3072)
    parser.add_argument(
        "--block_size", type=int, default=16, help="Block size for DFlash parallel prediction"
    )
    parser.add_argument(
        "--num_proc", type=int, default=16, help="Workers for tokenization (dataset.map)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Samples per forward pass (keep at 1 for variable-length seqs)",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument(
        "--sample_num", type=int, default=None, help="Limit number of samples (for debugging)"
    )
    parser.add_argument(
        "--min_loss_tokens",
        type=int,
        default=None,
        help="Minimum assistant-label tokens per cached example.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=0,
        help="Save a new sub-directory every N files (0 = no sharding)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sample indices whose cache files already exist.",
    )
    parser.add_argument(
        "--max_cache_tokens",
        type=int,
        default=None,
        help="Skip cache generation for sequences longer than this many tokens.",
    )

    return parser.parse_args()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def get_global_rank():
    return int(os.environ.get("RANK", 0))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def init_distributed():
    if get_world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)


def main():
    args = parse_args()
    init_distributed()

    rank = get_global_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    # --------------------------------------------------------------------------
    # 1. Load draft-model config (to get target_layer_ids)
    # --------------------------------------------------------------------------
    rank0_print("Loading draft model config...")
    draft_model_config = DraftModelConfig.from_file(args.draft_model_config_path)
    target_model_type = getattr(draft_model_config, "target_model_type", None)
    dflash_config = (
        getattr(draft_model_config, "dflash_config", None)
        or getattr(draft_model_config, "dflare_config", None)
        or {}
    )
    target_layer_ids = dflash_config.get("target_layer_ids", None)
    if target_layer_ids is None:
        raise ValueError(
            "dflash_config.target_layer_ids not found in draft_model_config. "
            f"Please set it in {args.draft_model_config_path}"
        )
    rank0_print(f"DFlash target layer IDs: {target_layer_ids}")

    # --------------------------------------------------------------------------
    # 2. Load target model (on this rank's GPU)
    # --------------------------------------------------------------------------
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    rank0_print("Loading target model...")
    target_extra_kwargs = {}
    if args.target_backend == "vllm":
        target_extra_kwargs = {
            # Hook-based extraction generates one token after prefill, so the
            # engine needs one slot beyond the dataset truncation length.
            "max_model_len": args.model_max_length + 1,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
            "max_num_seqs": args.max_num_seqs,
            "limit_mm_per_prompt": (
                {"image": 1, "video": 0}
                if target_model_type == "gemma4"
                else {"image": 0, "video": 0}
            ),
        }
    target_model = create_target_model(
        backend=args.target_backend,
        model_path=args.target_model_name_or_path,
        modal_type=(
            "VLM"
            if args.target_backend == "vllm" and target_model_type == "gemma4"
            else ("Gemma4" if target_model_type == "gemma4" else "LLM")
        ),
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        target_model_type=target_model_type,
        **target_extra_kwargs,
    )
    rank0_print("Target model loaded successfully")

    # --------------------------------------------------------------------------
    # 3. Tokenize dataset (using DatasetManager, same as online training)
    # --------------------------------------------------------------------------
    rank0_print("Building dataset...")
    # Temporarily patch args so DatasetManager picks the correct builder
    args.modal_type = "LLM"  # DFlash uses the LLM tokenisation path
    args.training_mode = "online"  # We want the text→token builder, not offline .ckpt loader

    dataset_manager = DatasetManager(
        data_args=args,
        tokenizer=target_model.tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
        target_model_type=target_model_type,
    )

    # Restore modal_type to DFlash so DFlash-specific filtering (min_loss_tokens) applies
    args.modal_type = "DFlash"

    (
        _,  # offline_train_dataset  (unused here)
        _,  # offline_eval_dataset
        online_train_dataset,
        _,  # online_eval_dataset
        _,  # data_collator
    ) = dataset_manager.create_all_datasets()

    if online_train_dataset is None:
        raise RuntimeError("No training dataset was created. Check --train_data_path.")

    rank0_print(f"Dataset size: {len(online_train_dataset)}")

    # --------------------------------------------------------------------------
    # 4. Distributed sampler: each rank processes its own shard
    # --------------------------------------------------------------------------
    sampler = DistributedSampler(
        online_train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    def collate_fn(batch):
        """Simple collate: each element is already a dict of 1-D tensors."""
        result = {}
        for key in batch[0]:
            tensors = [item[key] for item in batch]
            # Tensors may be 1-D or 2-D ([1, S]) — keep original shape
            try:
                result[key] = torch.stack(tensors)
            except Exception:
                result[key] = tensors
        return result

    dataloader = DataLoader(
        online_train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------------------------
    # 5. Output directory setup
    # --------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank0_print(f"Saving .ckpt files to: {output_dir}")
    rank0_print(f"World size={world_size}, this rank={rank}")

    # --------------------------------------------------------------------------
    # 6. Main loop: forward target model, save hidden states
    # --------------------------------------------------------------------------
    global_idx = 0  # index within this rank's portion
    skipped_nonfinite = 0
    skipped_too_long = 0
    total = len(dataloader)
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        # Shape may be [B, 1, S] or [B, S] depending on how dataset stores it
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(1)  # → [B, S]

        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(1)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(1)

        batch_sample_indices = [
            rank * len(dataloader) + global_idx + i
            for i in range(input_ids.size(0))
        ]

        def cache_path(sample_idx):
            save_dir = (
                output_dir / f"shard_{sample_idx // args.shard_size:05d}"
                if args.shard_size > 0
                else output_dir
            )
            return save_dir / f"sample_{sample_idx:08d}_rank{rank}.ckpt"

        if args.resume and all(cache_path(idx).exists() for idx in batch_sample_indices):
            global_idx += input_ids.size(0)
            continue
        if args.max_cache_tokens is not None and input_ids.shape[-1] > args.max_cache_tokens:
            skipped_too_long += input_ids.size(0)
            global_idx += input_ids.size(0)
            continue

        input_ids = input_ids.to(f"cuda:{local_rank}")
        attention_mask = attention_mask.to(f"cuda:{local_rank}")

        # Run target model
        hidden_states = target_model.get_hidden_states_only(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aux_hidden_states_layer_ids=target_layer_ids,
        )
        # hidden_states: [B, S, D*len(target_layer_ids)]

        # Save one .ckpt per sample in the batch
        for i in range(input_ids.size(0)):
            # Use a globally unique name: rank × position within rank
            sample_idx = rank * len(dataloader) + global_idx
            global_idx += 1

            ckpt_path = cache_path(sample_idx)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            sample_hidden_states = hidden_states[i : i + 1]
            if not torch.isfinite(sample_hidden_states).all():
                skipped_nonfinite += 1
                rank0_print(
                    f"[rank {rank}] Skipping sample {sample_idx}: "
                    "target hidden states contain NaN or Inf"
                )
                continue

            ckpt = {
                "input_ids": input_ids[i : i + 1].cpu(),  # [1, S]
                "hidden_states": sample_hidden_states.cpu().to(torch.bfloat16),  # [1, S, D*L]
                "loss_mask": loss_mask[i : i + 1].cpu(),  # [1, S]
                "attention_mask": attention_mask[i : i + 1].cpu(),  # [1, S]
            }
            torch.save(ckpt, ckpt_path)

        # Progress log
        if batch_idx % 100 == 0:
            elapsed = time.time() - t0
            samples_done = (batch_idx + 1) * args.batch_size
            speed = samples_done / elapsed if elapsed > 0 else 0
            rank0_print(
                f"[rank {rank}] {batch_idx + 1}/{total} batches | "
                f"{speed:.1f} samples/s | elapsed {elapsed:.0f}s"
            )

    if world_size > 1:
        dist.barrier()

    rank0_print(
        f"Data generation complete. Saved files to {output_dir}. "
        f"Skipped non-finite samples: {skipped_nonfinite}. "
        f"Skipped overlength samples: {skipped_too_long}"
    )


if __name__ == "__main__":
    main()
