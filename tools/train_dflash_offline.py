#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# DFlash offline training script.
# Trains a DFlash draft model using pre-computed hidden states (.ckpt files).
#
# Workflow:
#   Step 1 (data generation):
#       bash scripts/speculative/generate_qwen3_dflash_data.sh
#   Step 2 (offline training, this script):
#       bash scripts/speculative/run_qwen3_dflash_offline.sh

import argparse
import json
import os
from pathlib import Path

import torch

# Some ROCm torchvision wheels omit their optional C++ detection operators,
# while torchvision still registers fake implementations for them at import
# time. Declare the schemas before importing Transformers/torchvision so
# offline VLM training can start without requiring unused detection kernels.
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

import transformers

from angelslim.compressor.speculative import (
    DraftModelConfig,
    Eagle3TrainerFactory,
    create_draft_model,
    get_supported_chat_template_type_strings,
)
from angelslim.compressor.speculative.train.data.data_utils import (
    DataCollatorWithPadding,
)
from angelslim.compressor.speculative.train.data.dataset_builder.offline_dataset_builder import (
    OfflineEagle3Dataset,
)
from angelslim.compressor.speculative.train.models.draft.draft_model_factory import (
    DraftModelFactory,
)
from angelslim.utils import rank0_print

# ---------------------------------------------------------------------------
# Offline DFlash Dataset
# ---------------------------------------------------------------------------


class OfflineDFlashDataset(OfflineEagle3Dataset):
    """
    DFlash variant of the offline dataset.

    Each .ckpt file must contain:
        - input_ids:      LongTensor  [1, S]
        - hidden_states:  BFloat16Tensor [1, S, D*num_target_layers]   ← multi-layer hidden states
        - loss_mask:      LongTensor  [1, S]
        - attention_mask: LongTensor  [1, S]   (auto-generated if missing)

    Note: DFlash does NOT need target_hiddens (only Eagle3 offline uses the
    single final-layer hidden state). The multi-layer hidden_states is the
    context feature passed directly to the DFlash cross-attention.
    """

    REQUIRED_KEYS = ["input_ids", "hidden_states", "loss_mask"]

    def _load_ckpt(self, idx: int):
        import warnings

        ckpt_path = self.ckpt_files[idx]
        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            warnings.warn(
                f"Failed to load {ckpt_path}: {e}. Skipping.", RuntimeWarning, stacklevel=2
            )
            return None

        missing = [k for k in self.REQUIRED_KEYS if k not in data]
        if missing:
            warnings.warn(
                f"{ckpt_path} missing keys {missing}. Skipping.", RuntimeWarning, stacklevel=2
            )
            return None

        # Auto-generate attention_mask if absent
        if "attention_mask" not in data:
            data["attention_mask"] = torch.ones_like(data["input_ids"])

        return data


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash draft model (offline mode)")

    # Model
    m = parser.add_argument_group("Model Arguments")
    m.add_argument("--target_model_name_or_path", type=str, required=True)
    m.add_argument("--draft_model_config_path", type=str, required=True)
    m.add_argument(
        "--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    m.add_argument("--trust_remote_code", action="store_true", default=True)
    m.add_argument("--embed_weight_key", type=str, default=None)
    m.add_argument("--lm_head_key", type=str, default="lm_head.weight")
    m.add_argument(
        "--init_draft_checkpoint",
        type=str,
        default=None,
        help=(
            "Initialize draft-model weights from an AngelSlim save_pretrained "
            "checkpoint. This does not restore optimizer, scheduler, Trainer, "
            "or RNG state. An existing checkpoint-* under output_dir still "
            "takes precedence for interruption-safe resume."
        ),
    )
    m.add_argument(
        "--front_teacher_draft_path",
        type=str,
        default=None,
        help="Frozen B16 AngelSlim checkpoint used for front-logit distillation.",
    )
    m.add_argument("--front_teacher_block_size", type=int, default=16)
    m.add_argument("--front_distill_positions", type=int, default=15)
    m.add_argument("--front_distill_weight", type=float, default=0.0)
    m.add_argument("--front_distill_temperature", type=float, default=1.0)
    m.add_argument(
        "--train_tail_adapters_only",
        action="store_true",
        help="Freeze the DFlare backbone and optimize only tail_adapter_* parameters.",
    )
    m.add_argument(
        "--draft_arch",
        type=str,
        default=None,
        choices=["dflash", "dflare"],
        help=(
            "Override draft model architecture. If unset, uses the "
            "'architectures' field from the draft_model_config JSON. "
            "'dflash' -> QwenDFlashDraftModel, 'dflare' -> QwenDFlareDraftModel."
        ),
    )

    # DFlash-specific (override values in config JSON)
    d = parser.add_argument_group("DFlash Arguments")
    d.add_argument("--block_size", type=int, default=None)
    d.add_argument("--num_anchors", type=int, default=None)
    d.add_argument("--loss_decay_gamma", type=float, default=None)
    d.add_argument(
        "--horizon_boundary",
        type=int,
        default=None,
        help="First block offset receiving the horizon-tail loss weight.",
    )
    d.add_argument(
        "--horizon_tail_weight",
        type=float,
        default=None,
        help="Cross-entropy multiplier at and beyond --horizon_boundary.",
    )
    d.add_argument(
        "--acceptance_aware_loss",
        action="store_true",
        help=(
            "Use D-PACE dynamic token-CE weights derived from expected "
            "accepted-prefix length."
        ),
    )
    d.add_argument(
        "--dpace_alpha",
        type=float,
        default=None,
        help=(
            "D-PACE confidence-smoothing coefficient in [0, 1]. "
            "Defaults to 0.5 when --acceptance_aware_loss is enabled."
        ),
    )
    d.add_argument(
        "--hard_anchor_fraction",
        type=float,
        default=None,
        help=(
            "Fraction of sampled anchors relocated to the position where their "
            "block would be rejected, matching the anchor distribution serving "
            "visits. Serving blocks at a free anchor commit 6.84 tokens against "
            "3.27 at a rejection anchor, and rejection anchors are 95%% of "
            "steps, so uniform anchors train the wrong distribution. Costs one "
            "extra no-grad forward per step. 0 disables (default)."
        ),
    )
    d.add_argument("--mask_token_id", type=int, default=None)
    d.add_argument(
        "--attention_backend", type=str, default=None, choices=["flex_attention", "sdpa", "eager"]
    )

    # Data
    da = parser.add_argument_group("Data Arguments")
    da.add_argument(
        "--train_hidden_path",
        type=str,
        required=True,
        help="Directory of pre-computed training .ckpt files",
    )
    da.add_argument(
        "--eval_hidden_path",
        type=str,
        default=None,
        help="Directory of pre-computed eval .ckpt files (optional)",
    )
    da.add_argument(
        "--chat_template_type",
        type=str,
        default="qwen3",
        help=f"Supported: {', '.join(get_supported_chat_template_type_strings())}",
    )
    da.add_argument("--model_max_length", type=int, default=3072)
    da.add_argument("--num_proc", type=int, default=16)
    da.add_argument(
        "--cache_in_memory",
        action="store_true",
        default=False,
        help="Cache all .ckpt files in RAM (fast but memory-intensive)",
    )

    # Training
    t = parser.add_argument_group("Training Arguments")
    t.add_argument("--output_dir", type=str, required=True)
    t.add_argument("--optim", type=str, default="adamw_torch")
    t.add_argument("--num_train_epochs", type=int, default=6)
    t.add_argument("--per_device_train_batch_size", type=int, default=2)
    t.add_argument("--per_device_eval_batch_size", type=int, default=2)
    t.add_argument("--gradient_accumulation_steps", type=int, default=1)
    t.add_argument("--learning_rate", type=float, default=6e-4)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--warmup_steps", type=int, default=0)
    t.add_argument("--warmup_ratio", type=float, default=0.04)
    t.add_argument("--max_grad_norm", type=float, default=1.0)
    t.add_argument("--logging_steps", type=int, default=50)
    t.add_argument("--save_steps", type=float, default=2500)
    t.add_argument("--save_total_limit", type=int, default=None)
    t.add_argument("--eval_steps", type=int, default=500)
    t.add_argument("--save_strategy", type=str, default="steps")
    t.add_argument("--eval_strategy", type=str, default="no")
    t.add_argument("--lr_scheduler_type", type=str, default="cosine")
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--bf16", action="store_true")
    t.add_argument("--deepspeed", type=str, default=None)
    t.add_argument(
        "--fsdp",
        type=str,
        default="",
        help="FSDP configuration string passed to TrainingArguments "
        "(e.g. 'shard_grad_op auto_wrap'). Empty disables FSDP.",
    )
    t.add_argument(
        "--fsdp_config",
        type=str,
        default=None,
        help="Path to FSDP config JSON file (consumed by TrainingArguments).",
    )
    t.add_argument(
        "--dataloader_drop_last",
        action="store_true",
        default=False,
        help=(
            "Drop last incomplete batch. Note: when using DFlash trainer this "
            "is forced True internally to avoid FSDP shape mismatches on the "
            "trailing batch."
        ),
    )
    t.add_argument(
        "--gamma_warmup",
        action="store_true",
        default=False,
        help=(
            "Enable gamma warmup. When set, loss_decay_gamma is increased "
            "per epoch as: gamma = loss_decay_gamma + gamma_warmup_step * epoch."
        ),
    )
    t.add_argument(
        "--gamma_warmup_step",
        type=float,
        default=0.5,
        help="Per-epoch increment for gamma warmup. Default 0.5.",
    )
    t.add_argument("--report_to", type=str, default="none")
    t.add_argument("--run_name", type=str, default=None)
    t.add_argument("--training_time_test_length", type=int, default=7)

    # WandB
    w = parser.add_argument_group("WandB Arguments")
    w.add_argument("--wandb_project", type=str, default=None)
    w.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def _setup_wandb(args):
    if args.report_to not in ("wandb", "all"):
        return
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    run_name = args.wandb_run_name or args.run_name or os.environ.get("WANDB_RUN_NAME")
    if run_name:
        os.environ["WANDB_RUN_NAME"] = run_name
        args.run_name = run_name
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        try:
            import wandb

            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "angelslim-dflash"),
                name=run_name,
                resume="allow",
            )
        except ImportError:
            print("[WARNING] wandb not installed. Install via: pip install wandb")


_CHECKPOINT_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_STRUCTURAL_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "target_hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_target_layers",
)


def _validate_init_draft_checkpoint(checkpoint, draft_model_config):
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_dir():
        raise ValueError(
            "--init_draft_checkpoint must be an existing AngelSlim checkpoint "
            f"directory; got: {checkpoint_path}"
        )

    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        raise ValueError(
            f"Draft initialization checkpoint is missing config.json: {checkpoint_path}"
        )
    if not any((checkpoint_path / filename).is_file() for filename in _CHECKPOINT_WEIGHT_FILES):
        expected = ", ".join(_CHECKPOINT_WEIGHT_FILES)
        raise ValueError(
            "Draft initialization checkpoint has no supported model weights "
            f"({expected}): {checkpoint_path}"
        )

    try:
        source_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read checkpoint config {config_path}: {exc}") from exc

    expected_architectures = list(getattr(draft_model_config, "architectures", []) or [])
    source_architectures = list(source_config.get("architectures", []) or [])
    if source_architectures != expected_architectures:
        raise ValueError(
            "Draft initialization architecture mismatch: checkpoint has "
            f"{source_architectures}, requested config has {expected_architectures}"
        )

    mismatches = []
    for field in _STRUCTURAL_CONFIG_FIELDS:
        expected = getattr(draft_model_config, field, None)
        actual = source_config.get(field)
        if expected is not None and actual is not None and actual != expected:
            mismatches.append(f"{field}: checkpoint={actual!r}, requested={expected!r}")
    if mismatches:
        raise ValueError(
            "Draft initialization checkpoint is structurally incompatible: "
            + "; ".join(mismatches)
        )

    return checkpoint_path


def _load_initialized_draft_model(draft_model_config, checkpoint):
    checkpoint_path = _validate_init_draft_checkpoint(checkpoint, draft_model_config)
    model_class = DraftModelFactory._get_model_class(draft_model_config)
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    config_dtype = getattr(draft_model_config, "torch_dtype", None)
    model_dtype = dtype_mapping.get(str(config_dtype), config_dtype)

    model, loading_info = model_class.from_pretrained(
        checkpoint_path,
        config=draft_model_config,
        dtype=model_dtype,
        local_files_only=True,
        output_loading_info=True,
    )
    load_problems = {
        key: loading_info.get(key, [])
        for key in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if loading_info.get(key)
    }
    if load_problems:
        details = "; ".join(f"{key}={value}" for key, value in load_problems.items())
        raise ValueError(
            "Draft initialization checkpoint did not load exactly into the "
            f"requested model: {details}"
        )
    return model


def _has_trainer_checkpoint(output_dir):
    return any(Path(output_dir).glob("checkpoint-*"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train():
    args = parse_args()
    _setup_wandb(args)
    resume_existing_run = _has_trainer_checkpoint(args.output_dir)

    # dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    # torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    # ------------------------------------------------------------------
    # 1. Draft model config
    # ------------------------------------------------------------------
    rank0_print("Loading draft model config...")
    draft_model_config = DraftModelConfig.from_file(args.draft_model_config_path)
    target_model_type = getattr(draft_model_config, "target_model_type", None)
    draft_model_config.target_model_name_or_path = args.target_model_name_or_path
    if args.embed_weight_key is not None:
        draft_model_config.embed_weight_key = args.embed_weight_key
    draft_model_config.trust_remote_code = args.trust_remote_code

    # Optionally override draft architecture from CLI. Both DFlash and DFlare
    # share the same Qwen3Config schema (block_size, dflash_config, etc.), so
    # swapping the architectures field is sufficient to route create_draft_model
    # to the desired class.
    if args.draft_arch is not None:
        arch_map = {
            "dflash": "QwenDFlashDraftModel",
            "dflare": (
                "Gemma4DFlareDraftModel"
                if target_model_type == "gemma4"
                else "QwenDFlareDraftModel"
            ),
        }
        new_arch = arch_map[args.draft_arch]
        rank0_print(
            f"Overriding draft architecture: "
            f"{getattr(draft_model_config, 'architectures', None)} -> [{new_arch}]"
        )
        draft_model_config.architectures = [new_arch]

    # Override DFlash params from CLI if specified
    if args.block_size is not None:
        draft_model_config.block_size = args.block_size
    if args.num_anchors is not None:
        draft_model_config.num_anchors = args.num_anchors
    if args.loss_decay_gamma is not None:
        draft_model_config.loss_decay_gamma = args.loss_decay_gamma
    if args.horizon_boundary is not None:
        draft_model_config.horizon_boundary = args.horizon_boundary
    if args.horizon_tail_weight is not None:
        draft_model_config.horizon_tail_weight = args.horizon_tail_weight
    if args.acceptance_aware_loss:
        draft_model_config.acceptance_aware_loss = True
    if args.dpace_alpha is not None:
        if not 0.0 <= args.dpace_alpha <= 1.0:
            raise ValueError(
                f"--dpace_alpha must be in [0, 1], got {args.dpace_alpha}"
            )
        draft_model_config.dpace_alpha = args.dpace_alpha
    if args.hard_anchor_fraction is not None:
        draft_model_config.hard_anchor_fraction = args.hard_anchor_fraction
    draft_model_config.front_teacher_block_size = args.front_teacher_block_size
    draft_model_config.front_distill_positions = args.front_distill_positions
    draft_model_config.front_distill_weight = args.front_distill_weight
    draft_model_config.front_distill_temperature = args.front_distill_temperature
    draft_model_config.train_tail_adapters_only = args.train_tail_adapters_only
    # Always propagate gamma_warmup flags to the draft model config so the
    # trainer can pick them up regardless of CLI defaults.
    draft_model_config.gamma_warmup = args.gamma_warmup
    draft_model_config.gamma_warmup_step = args.gamma_warmup_step
    if args.attention_backend is not None:
        draft_model_config.attention_backend = args.attention_backend
        draft_model_config._attn_implementation = args.attention_backend
    if args.mask_token_id is not None:
        dfc = (
            getattr(draft_model_config, "dflash_config", None)
            or getattr(draft_model_config, "dflare_config", None)
            or {}
        )
        dfc["mask_token_id"] = args.mask_token_id
        draft_model_config.dflash_config = dfc

    # ------------------------------------------------------------------
    # 2. Draft model
    # ------------------------------------------------------------------
    if args.init_draft_checkpoint and not resume_existing_run:
        rank0_print(
            "Initializing draft model weights from AngelSlim checkpoint: "
            f"{args.init_draft_checkpoint}"
        )
        draft_model = _load_initialized_draft_model(
            draft_model_config,
            args.init_draft_checkpoint,
        )
        rank0_print(
            "Loaded draft weights only; optimizer, scheduler, Trainer, and RNG "
            "state will start fresh."
        )
    else:
        if args.init_draft_checkpoint:
            rank0_print(
                "Found checkpoint-* in output_dir; resuming that interrupted "
                "run instead of applying --init_draft_checkpoint."
            )
        else:
            rank0_print("Loading draft model...")
        draft_model = create_draft_model(draft_model_config)
    rank0_print(f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}")
    front_teacher_model = None
    if args.front_teacher_draft_path is not None:
        teacher_path = Path(args.front_teacher_draft_path)
        teacher_config = DraftModelConfig.from_file(teacher_path / "config.json")
        front_teacher_model = _load_initialized_draft_model(
            teacher_config,
            teacher_path,
        )
        front_teacher_model.eval()
        front_teacher_model.requires_grad_(False)
        rank0_print(
            "Loaded frozen front teacher from "
            f"{teacher_path} at B{teacher_config.block_size}"
        )

    # ------------------------------------------------------------------
    # 3. Offline datasets
    # ------------------------------------------------------------------
    rank0_print(f"Loading offline training data from: {args.train_hidden_path}")
    train_dataset = OfflineDFlashDataset(
        data_dir=args.train_hidden_path,
        file_pattern="*.ckpt",
        cache_in_memory=args.cache_in_memory,
    )
    rank0_print(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    if args.eval_hidden_path:
        rank0_print(f"Loading offline eval data from: {args.eval_hidden_path}")
        eval_dataset = OfflineDFlashDataset(
            data_dir=args.eval_hidden_path,
            file_pattern="*.ckpt",
            cache_in_memory=args.cache_in_memory,
        )
        rank0_print(f"Eval samples: {len(eval_dataset)}")

    data_collator = DataCollatorWithPadding()

    # ------------------------------------------------------------------
    # 4. TrainingArguments
    # ------------------------------------------------------------------
    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=args.run_name,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp,
        # Force drop_last=True (AngelSlim default) to avoid FSDP shape
        # mismatches on the trailing batch.
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )
    if args.fsdp_config:
        ta_kwargs["fsdp_config"] = args.fsdp_config

    training_args = transformers.TrainingArguments(**ta_kwargs)

    # ------------------------------------------------------------------
    # 5. Trainer -- use Eagle3TrainerFactory
    # ------------------------------------------------------------------
    rank0_print("Initializing trainer...")
    trainer = Eagle3TrainerFactory.create(
        training_mode="offline",
        modal_type="DFlash",
        draft_model=draft_model,
        target_model=None,  # Not needed — hidden states are pre-computed
        length=args.training_time_test_length,
        draft_model_config=draft_model_config,
        front_teacher_model=front_teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    if resume_existing_run:
        rank0_print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting fresh training run...")
        trainer.train()

    trainer.save_model(training_args.output_dir)
    rank0_print(f"Saved final draft model to {training_args.output_dir}")
    rank0_print("Training completed!")


if __name__ == "__main__":
    train()
