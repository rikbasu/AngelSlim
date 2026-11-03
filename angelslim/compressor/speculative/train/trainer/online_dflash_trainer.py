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

"""Online DFlash Trainer for speculative decoding training.

DFlash uses block-parallel cross-attention rather than Eagle3's
iterative autoregressive approach, so it overrides compute_loss
with its own block-wise CE loss logic.
"""

import gc
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig

from .dflash_position_metrics import (
    dflash_position_metric_counts,
    empty_dflash_position_counts,
    finalize_dflash_position_metrics,
)
from .eagle3_trainer import Eagle3Trainer
from .trainer_factory import Eagle3TrainerFactory

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def acceptance_aware_position_weights(
    token_nll: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute D-PACE weights for expected accepted-prefix length.

    D-PACE smooths detached target-token probabilities only inside the
    position weights: q_tilde = (1 - alpha) * q + alpha. This prevents
    cumulative products, and therefore later-position weights, from
    vanishing. The token cross-entropy itself remains unsmoothed.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"D-PACE alpha must be in [0, 1], got {alpha}")

    valid = valid_mask.bool()
    correct_prob = torch.exp(-token_nll.detach()).clamp_(min=1e-6, max=1.0)
    smoothed_prob = (1.0 - alpha) * correct_prob + alpha
    step_prob = torch.where(valid, smoothed_prob, torch.ones_like(smoothed_prob))
    prefix_survival = torch.cumprod(step_prob, dim=-1) * valid.float()
    downstream_value = torch.flip(
        torch.cumsum(torch.flip(prefix_survival, dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    normalizer = (downstream_value * valid.float()).sum() / valid.float().sum().clamp_min(
        1.0
    )
    return torch.where(
        valid,
        downstream_value / normalizer.clamp_min(1e-6),
        torch.zeros_like(downstream_value),
    )


def front_distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    num_positions: int,
    temperature: float = 1.0,
    position_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward KL from a B16 teacher on front positions only."""
    if temperature <= 0:
        raise ValueError(
            f"front distillation temperature must be positive, got {temperature}"
        )
    limit = min(
        int(num_positions) + 1,
        student_logits.shape[-2],
        teacher_logits.shape[-2],
    )
    if limit <= 1:
        return student_logits.sum() * 0.0

    student = student_logits[..., 1:limit, :].float() / temperature
    teacher = teacher_logits[..., 1:limit, :].float() / temperature
    teacher_log_probs = F.log_softmax(teacher, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student, dim=-1)
    per_position = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1) * (temperature**2)

    weights = valid_mask[..., 1:limit].float()
    if position_weights is not None:
        weights = weights * position_weights[..., 1:limit].float()
    return (per_position * weights).sum() / weights.sum().clamp_min(1e-6)


def freeze_front_teacher_model(model: nn.Module) -> nn.Module:
    """Put a front-distillation teacher in inference-only mode."""
    model.eval()
    model.requires_grad_(False)
    return model


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    causal_within_block: bool = False,
    attention_backend: str = "flex_attention",
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        if causal_within_block:
            q_offset = q_idx % block_size
            kv_offset = (kv_idx - S) % block_size
            mask_draft = mask_draft & (kv_offset <= q_offset)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    if attention_backend != "flex_attention":
        q_idx = torch.arange(Q_LEN, device=device)
        kv_idx = torch.arange(KV_LEN, device=device)
        q_block_id = q_idx // block_size
        anchors = anchor_positions[:, q_block_id]

        is_context = kv_idx < S
        mask_context = is_context[None, None, :] & (
            kv_idx[None, None, :] < anchors[:, :, None]
        )
        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S).clamp_min(0) // block_size
        mask_draft = is_draft[None, None, :] & (
            q_block_id[None, :, None] == kv_block_id[None, None, :]
        )
        if causal_within_block:
            q_offset = q_idx % block_size
            kv_offset = (kv_idx - S).clamp_min(0) % block_size
            mask_draft &= (
                kv_offset[None, None, :] <= q_offset[None, :, None]
            )
        valid = block_keep_mask[:, q_block_id, None]
        return ((mask_context | mask_draft) & valid).unsqueeze(1)

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class TargetEmbeddingsAndHead(nn.Module):
    """Efficiently loads only the embedding layer and lm_head from a pretrained model.

    Handles safetensors slicing and Weight Tying correctly.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        text_config = (
            config.get_text_config()
            if hasattr(config, "get_text_config")
            else getattr(config, "text_config", config)
        )
        vocab_size = getattr(text_config, "vocab_size", getattr(config, "vocab_size", None))
        hidden_size = getattr(text_config, "hidden_size", getattr(config, "hidden_size", None))
        pad_token_id = getattr(text_config, "pad_token_id", None)

        self.embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=pad_token_id,
        )
        self.embedding_scale = (
            hidden_size**0.5 if getattr(config, "model_type", None) == "gemma4" else 1.0
        )
        self.logit_softcap = getattr(text_config, "final_logit_softcapping", None)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetEmbeddingsAndHead":

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        instance = cls(config)

        if embed_key is None:
            model_type = getattr(config, "model_type", None)
            embed_key = (
                "model.language_model.embed_tokens.weight"
                if model_type == "gemma4"
                else "model.embed_tokens.weight"
            )
        if lm_head_key is None:
            lm_head_key = "lm_head.weight"

        tie_weights = getattr(config, "tie_word_embeddings", False)
        instance._load_weights(model_path, embed_key, lm_head_key, tie_weights)

        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(self, model_path: str, embed_key: str, lm_head_key: str, tie_weights: bool):
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        files_to_load = {}

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            if embed_key in weight_map:
                files_to_load[embed_key] = weight_map[embed_key]
            else:
                raise ValueError(f"Embedding key '{embed_key}' not found in weight map.")

            if not tie_weights:
                if lm_head_key in weight_map:
                    files_to_load[lm_head_key] = weight_map[lm_head_key]
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)

            if not target_file:
                raise FileNotFoundError("No checkpoint found.")

            files_to_load[embed_key] = os.path.basename(target_file)
            if not tie_weights:
                files_to_load[lm_head_key] = os.path.basename(target_file)

        file_to_keys_map = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            if full_path not in file_to_keys_map:
                file_to_keys_map[full_path] = []
            file_to_keys_map[full_path].append(key)

        for file_path, keys in file_to_keys_map.items():
            self._load_file_content(file_path, keys, embed_key, lm_head_key)

        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

    def _load_file_content(
        self,
        file_path: str,
        keys_to_extract: list,
        target_embed_key: str,
        target_head_key: str,
    ):
        state_dict_part = {}

        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys_to_extract:
                    if k in f.keys():
                        state_dict_part[k] = f.get_tensor(k)
        else:
            full_state = torch.load(file_path, map_location="cpu")
            for k in keys_to_extract:
                if k in full_state:
                    state_dict_part[k] = full_state[k]
            del full_state
            gc.collect()

        for k, tensor in state_dict_part.items():
            if k == target_embed_key:
                self.embed_tokens.weight.data.copy_(tensor)
            elif k == target_head_key:
                if tensor.shape == self.lm_head.weight.data.shape:
                    self.lm_head.weight.data.copy_(tensor)


class _FP32StateAdamW(torch.optim.Optimizer):
    """AdamW with fp32 master weights (AngelSlim DFlash optimizer).

    Maintains fp32 master copies of all parameters (in optimizer state).
    On each step:
      1. Cast bf16 gradients to fp32.
      2. Clip fp32 grad norm.
      3. Adam update on fp32 master weights.
      4. Copy fp32 master -> bf16 model params.

    Key properties:
      * Accumulation in fp32 (no precision loss from bf16 quantization).
      * Grad clipping on fp32 gradients.
      * Only the final copy-back introduces bf16 quantization.

    Compatible with FSDP + accelerate + HF Trainer (operates on the SAME
    parameter objects required for FSDP state_dict).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        super().__init__(params, defaults)

        # Eagerly initialize all master parameters at construction so all
        # ranks start from synchronized bf16 params before any training step.
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    self._initialize_param_state(p)

    def _initialize_param_state(self, param: torch.Tensor) -> dict:
        """Initialize state for original or FSDP-rewrapped parameters."""
        state = self.state[param]
        if "exp_avg" not in state:
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            state["master_param"] = param.data.detach().clone().to(torch.float32)
        return state

    def _prune_stale_param_state(self) -> None:
        """Drop state keyed by parameters replaced during FSDP2 wrapping."""
        active_params = {
            param
            for group in self.param_groups
            for param in group["params"]
        }
        for param in list(self.state):
            if isinstance(param, torch.Tensor) and param not in active_params:
                del self.state[param]

    @torch.no_grad()
    def step(self, closure=None):
        """Full fp32 master-weight update step (AngelSlim DFlash optimizer)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._prune_stale_param_state()

        # Phase 1: Cast all bf16 grads to fp32 (kept temporarily in state).
        all_fp32_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self._initialize_param_state(p)

                # Ensure states are fp32 (handles resume from checkpoint).
                if state["exp_avg"].dtype != torch.float32:
                    state["exp_avg"] = state["exp_avg"].to(torch.float32)
                if state["exp_avg_sq"].dtype != torch.float32:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(torch.float32)
                if state["master_param"].dtype != torch.float32:
                    state["master_param"] = state["master_param"].to(torch.float32)

                fp32_grad = p.grad.detach().to(torch.float32)
                state["_fp32_grad"] = fp32_grad
                all_fp32_grads.append(fp32_grad)

        # Phase 2: Clip fp32 grad norm.
        # Manual clipping because all_fp32_grads holds plain tensors (not Parameters).
        # In FSDP SHARD_GRAD_OP + use_orig_params=True, p.grad is the full
        # all-reduced gradient (same on all ranks), so per-rank clipping is correct.
        if self.max_grad_norm > 0 and all_fp32_grads:
            total_norm_sq = sum(g.norm().pow(2) for g in all_fp32_grads)
            total_norm = total_norm_sq.sqrt()
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            clip_coef_clamped = min(clip_coef.item(), 1.0)
            if clip_coef_clamped < 1.0:
                for g in all_fp32_grads:
                    g.mul_(clip_coef_clamped)

        # Phase 3: Adam update on fp32 master weights, then copy back to bf16.
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self._initialize_param_state(p)
                grad = state.pop("_fp32_grad")

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                master_param = state["master_param"]

                state["step"] += 1
                step_t = state["step"].item()

                # Decoupled weight decay on fp32 master (AdamW style).
                if weight_decay != 0:
                    master_param.mul_(1.0 - lr * weight_decay)

                # Adam update in fp32.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step_t
                bias_correction2 = 1 - beta2**step_t

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)

                master_param.addcdiv_(exp_avg, denom, value=-step_size)

                # Copy fp32 master -> bf16 model param (only quantization point).
                p.data.copy_(master_param.to(p.dtype))
                p.grad = None

        return loss

    def state_dict(self):
        self._prune_stale_param_state()
        return super().state_dict()


class _FP32MasterWeightOptimizer(torch.optim.Optimizer):
    """Thin wrapper around any torch optimizer that maintains fp32 master weights.

    AngelSlim DFlash fp32 master-weight pattern:
      1. At construction, clone bf16 model params -> fp32 master copies.
      2. On every step():
         a. Copy bf16 grads -> fp32 master grads (cast up), clear bf16 grads.
         b. Clip fp32 grad norm.
         c. Run inner optimizer step on fp32 params.
         d. Copy updated fp32 values -> bf16 model params (cast down).

    Used as ``self.optimizer`` inside HF Trainer for the DDP code path.
    HF Trainer calls ``self.optimizer.step()`` / ``self.optimizer.zero_grad()``
    directly, so placing the sync logic inside the optimizer is the only
    reliable way to ensure it actually runs.

    Inherits from torch.optim.Optimizer so isinstance checks in HF Trainer's
    lr_scheduler creation (LambdaLR.__init__) pass correctly.
    """

    def __init__(
        self,
        bf16_params: List[torch.Tensor],
        inner_optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
    ):
        self._bf16_params = bf16_params

        # Build fp32 master copies and replace the optimizer's param groups.
        self._fp32_params: List[torch.Tensor] = [
            p.detach().clone().to(torch.float32).requires_grad_(True) for p in bf16_params
        ]
        assert len(inner_optimizer.param_groups) == 1, (
            "_FP32MasterWeightOptimizer expects a single param group; "
            "extend if/when multiple groups (e.g. LoRA, lr split) are needed."
        )
        inner_optimizer.param_groups[0]["params"] = self._fp32_params
        # Re-initialise state dict for the new param objects.
        from collections import defaultdict

        inner_optimizer.state = defaultdict(dict)

        self._inner = inner_optimizer
        self.max_grad_norm = max_grad_norm

        # Call torch.optim.Optimizer.__init__ so that isinstance(self, Optimizer)
        # returns True. The _initializing flag prevents add_param_group from
        # delegating to self._inner during super().__init__ (which would
        # create a duplicate param group in the inner optimizer).
        self._initializing = True
        super().__init__(self._fp32_params, inner_optimizer.defaults)
        self._initializing = False

        # Redirect param_groups and state to inner optimizer's versions so
        # lr_scheduler / lr logging always see the correct param groups.
        self.param_groups = self._inner.param_groups
        self.state = self._inner.state

    # ------------------------------------------------------------------ #
    # Core step / zero_grad — called directly by HF Trainer               #
    # ------------------------------------------------------------------ #

    def step(self, closure=None):
        """Full fp32 master-weight update step."""
        with torch.no_grad():
            # (a) Copy bf16 grads -> fp32 master grads.
            for bf16_p, fp32_p in zip(self._bf16_params, self._fp32_params):
                if bf16_p.grad is not None:
                    fp32_p.grad = bf16_p.grad.detach().to(torch.float32)
                    bf16_p.grad = None
                else:
                    fp32_p.grad = None

            # (b) Clip fp32 grad norm.
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._fp32_params, self.max_grad_norm)

        # (c) Optimizer step on fp32 params.
        loss = self._inner.step(closure)

        # (d) Copy fp32 -> bf16 model params.
        with torch.no_grad():
            for bf16_p, fp32_p in zip(self._bf16_params, self._fp32_params):
                bf16_p.data.copy_(fp32_p.data.to(bf16_p.dtype))

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients on both bf16 model params and fp32 master params."""
        for bf16_p in self._bf16_params:
            if set_to_none:
                bf16_p.grad = None
            elif bf16_p.grad is not None:
                bf16_p.grad.zero_()
        for fp32_p in self._fp32_params:
            if set_to_none:
                fp32_p.grad = None
            elif fp32_p.grad is not None:
                fp32_p.grad.zero_()

    # ------------------------------------------------------------------ #
    # Delegate everything else to the inner optimizer                      #
    # ------------------------------------------------------------------ #

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        # During super().__init__ the inner optimizer is not yet assigned,
        # so fall back to the default Optimizer behaviour.
        if getattr(self, "_initializing", True):
            return super().add_param_group(param_group)
        return self._inner.add_param_group(param_group)

    def __repr__(self):
        return f"_FP32MasterWeightOptimizer({self._inner})"


@Eagle3TrainerFactory.register("online", "DFlash")
class OnlineDFlashTrainer(Eagle3Trainer):
    """Online DFlash Trainer for speculative decoding training.

    Uses block-parallel cross-attention and anchor-based CE loss
    rather than Eagle3's iterative autoregressive training loop.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        length: int,
        draft_model_config: Dict[str, Any],
        front_teacher_model: nn.Module | None = None,
        **kwargs,
    ):
        """
        Initialize the OnlineDFlashTrainer.

        Args:
            draft_model: DFlash draft model
            target_model: Target model for generating hidden states
            length: Not used for DFlash (kept for interface compatibility)
            draft_model_config: Configuration dictionary for draft model,
                must contain dflash-specific fields
            **kwargs: Additional arguments passed to parent Trainer
        """
        train_tail_adapters_only = bool(
            getattr(draft_model_config, "train_tail_adapters_only", False)
        )
        if train_tail_adapters_only:
            trainable = []
            for name, parameter in draft_model.named_parameters():
                parameter.requires_grad_(name.startswith("tail_adapter_"))
                if parameter.requires_grad:
                    trainable.append(name)
            if not trainable:
                raise ValueError(
                    "train_tail_adapters_only requires configured tail adapters"
                )
        super().__init__(draft_model=draft_model, length=length, **kwargs)
        self.target_model = target_model
        self.train_tail_adapters_only = train_tail_adapters_only
        self._aux_hidden_states_layer_ids = getattr(
            draft_model_config, "aux_hidden_states_layer_ids", None
        )

        # Extract DFlash-specific config
        dflash_config = (
            getattr(draft_model_config, "dflash_config", None)
            or getattr(draft_model_config, "dflare_config", None)
            or {}
        )
        self.block_size = getattr(draft_model_config, "block_size", 16)
        self.num_anchors = getattr(draft_model_config, "num_anchors", 512)
        self.loss_decay_gamma = getattr(draft_model_config, "loss_decay_gamma", None)
        # Gamma warmup: gradually increase loss_decay_gamma per epoch
        # (AngelSlim DFlash gamma-warmup schedule).
        self._gamma_init = self.loss_decay_gamma
        self.gamma_warmup = getattr(draft_model_config, "gamma_warmup", False)
        self._gamma_step = getattr(draft_model_config, "gamma_warmup_step", 0.5)
        self.attention_backend = getattr(draft_model_config, "attention_backend", "flex_attention")
        # Fraction of anchors relocated onto the rejection-point distribution
        # that serving actually visits. Costs one extra no-grad forward.
        self.hard_anchor_fraction = float(
            getattr(draft_model_config, "hard_anchor_fraction", 0.0) or 0.0
        )
        self.max_train_positions = getattr(
            draft_model_config,
            "max_train_positions",
            None,
        )
        self.horizon_boundary = getattr(
            draft_model_config,
            "horizon_boundary",
            None,
        )
        self.horizon_tail_weight = float(
            getattr(draft_model_config, "horizon_tail_weight", 1.0) or 1.0
        )
        self.acceptance_aware_loss = bool(
            getattr(draft_model_config, "acceptance_aware_loss", False)
        )
        self.dpace_alpha = float(
            getattr(draft_model_config, "dpace_alpha", 0.5)
        )
        if not 0.0 <= self.dpace_alpha <= 1.0:
            raise ValueError(
                f"dpace_alpha must be in [0, 1], got {self.dpace_alpha}"
            )
        self.ar_aux_weight = float(
            getattr(draft_model_config, "ar_aux_weight", 0.0) or 0.0
        )
        self.front_distill_weight = float(
            getattr(draft_model_config, "front_distill_weight", 0.0) or 0.0
        )
        self.front_distill_positions = int(
            getattr(draft_model_config, "front_distill_positions", 15) or 15
        )
        self.front_distill_temperature = float(
            getattr(draft_model_config, "front_distill_temperature", 1.0) or 1.0
        )
        self.front_teacher_block_size = int(
            getattr(draft_model_config, "front_teacher_block_size", 16) or 16
        )
        self.front_teacher_model = front_teacher_model
        if self.front_distill_weight > 0 and self.front_teacher_model is None:
            raise ValueError(
                "front_teacher_model is required when front_distill_weight > 0"
            )
        if self.front_teacher_model is not None:
            freeze_front_teacher_model(self.front_teacher_model)
        self.mask_token_id = dflash_config.get(
            "mask_token_id",
            getattr(draft_model_config, "mask_token_id", None),
        )

        # Sync _attn_implementation on the draft model so its attention layers
        # dispatch to the correct backend (eager vs flex_attention vs sdpa).
        if self.attention_backend == "eager":
            draft_model.config._attn_implementation = "eager"
        elif self.attention_backend == "flex_attention":
            draft_model.config._attn_implementation = "flex_attention"
        else:
            draft_model.config._attn_implementation = self.attention_backend

        # fp32 master weights optimizer — set by create_optimizer() (DDP path).
        # FSDP path uses _FP32StateAdamW directly as self.optimizer instead.
        self._fp32_optimizer: Optional["_FP32MasterWeightOptimizer"] = None

        # Load target model's lm_head and embed_tokens
        # In offline mode target_model may be None; fall back to config path.
        target_model_path = None
        if target_model is not None:
            target_model_path = getattr(target_model, "model_path", None)
        if target_model_path is None:
            target_model_path = getattr(draft_model_config, "target_model_name_or_path", None)
        embed_weight_key = getattr(
            draft_model_config, "embed_weight_key", "model.embed_tokens.weight"
        )
        lm_head_key = getattr(draft_model_config, "lm_head_key", "lm_head.weight")
        trust_remote_code = getattr(draft_model_config, "trust_remote_code", True)

        if target_model_path is not None:
            target_components = TargetEmbeddingsAndHead.from_pretrained(
                target_model_path,
                embed_key=embed_weight_key,
                lm_head_key=lm_head_key,
                device="cuda",
                trust_remote_code=trust_remote_code,
            )
            self.target_lm_head = target_components.lm_head
            self.target_embed_tokens = target_components.embed_tokens
            self.target_embedding_scale = target_components.embedding_scale
            self.target_logit_softcap = target_components.logit_softcap
        else:
            raise ValueError(
                "target_model_name_or_path must be set in draft_model_config "
                "or target_model.model_path for DFlash training."
            )

    def create_optimizer(self, model=None):
        """Create optimizer for DFlash training.

        Three branches:
          * DeepSpeed: defer to HF Trainer's default optimizer creation.
          * FSDP: AdamW with fp32 optimizer states (``_FP32StateAdamW``),
            using the AngelSlim DFlash fp32-master pattern. Critical because
            bf16 momentum and variance only have 7-bit mantissa, which causes
            training quality degradation after a few thousand steps.
          * DDP / single GPU: ``_FP32MasterWeightOptimizer`` wrapping AdamW for
            fp32 master weight updates.
        """
        if self.is_deepspeed_enabled:
            return super().create_optimizer(model)

        if self.is_fsdp_enabled:
            args = self.args
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad]}]
            optimizer = _FP32StateAdamW(
                param_groups,
                lr=args.learning_rate,
                betas=(
                    getattr(args, "adam_beta1", 0.9),
                    getattr(args, "adam_beta2", 0.999),
                ),
                eps=getattr(args, "adam_epsilon", 1e-8),
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
            )
            self.optimizer = optimizer
            return self.optimizer

        bf16_params: List[torch.Tensor] = [p for p in self.model.parameters() if p.requires_grad]
        if not bf16_params:
            return super().create_optimizer(model)

        from torch.optim import AdamW

        args = self.args
        inner_optimizer = AdamW(
            # Placeholder — _FP32MasterWeightOptimizer will replace param_groups
            # with fp32 copies immediately after construction.
            bf16_params,
            lr=args.learning_rate,
            betas=(
                getattr(args, "adam_beta1", 0.9),
                getattr(args, "adam_beta2", 0.999),
            ),
            eps=getattr(args, "adam_epsilon", 1e-8),
            weight_decay=args.weight_decay,
        )

        fp32_opt = _FP32MasterWeightOptimizer(
            bf16_params=bf16_params,
            inner_optimizer=inner_optimizer,
            max_grad_norm=args.max_grad_norm,
        )
        self._fp32_optimizer = fp32_opt
        self.optimizer = fp32_opt
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create LR scheduler: AngelSlim DFlash CosineAnnealingWarmupLR.

        AngelSlim warmup formula:  lr = base_lr * (step + 1) / warmup_steps
        HF Trainer warmup formula: lr = base_lr * step / warmup_steps

        The +1 offset means step 0 yields lr = base_lr / warmup_steps instead
        of 0. After warmup, both use identical cosine annealing.
        """
        import math

        from torch.optim.lr_scheduler import LambdaLR

        if optimizer is None:
            optimizer = self.optimizer

        warmup_steps = self.args.get_warmup_steps(num_training_steps)

        def angelslim_cosine_schedule(current_step: int) -> float:
            """LR multiplier for AngelSlim DFlash CosineAnnealingWarmupLR."""
            if current_step < warmup_steps:
                # AngelSlim: (last_epoch + 1) / warmup_epochs * base_lr
                # After N step() calls last_epoch = N, so first lr = 1/warmup_steps.
                return float(current_step + 1) / float(max(1, warmup_steps))
            # Cosine decay phase — identical to HF.
            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.lr_scheduler = LambdaLR(optimizer, angelslim_cosine_schedule)
        return self.lr_scheduler

    def _clip_grad_norm(self, *args, **kwargs):
        """Skip HF Trainer's built-in grad clipping when running our fp32 optimizers.

        Both ``_FP32MasterWeightOptimizer`` (DDP) and ``_FP32StateAdamW`` (FSDP)
        clip gradients internally on fp32 grads (AngelSlim DFlash fp32-master
        clipping). HF Trainer's Accelerator-based clip_grad_norm_ would
        otherwise run on bf16 grads (incorrect precision for clipping), causing
        DOUBLE CLIPPING that significantly slows down training.
        """
        if self._fp32_optimizer is not None:
            # DDP path — clipped inside _FP32MasterWeightOptimizer.step()
            return torch.tensor(0.0)

        # FSDP path: _FP32StateAdamW clips internally on fp32 grads.
        # self.optimizer may be wrapped by AcceleratedOptimizer, so unwrap once.
        optimizer = self.optimizer
        if hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        if isinstance(optimizer, _FP32StateAdamW):
            return torch.tensor(0.0)

        return super()._clip_grad_norm(*args, **kwargs)

    def save_optimizer_and_scheduler(self, output_dir, **kwargs):
        """Override to handle fp32 master weight optimizer with FSDP.

        FSDP's built-in optim_state_dict() cannot handle our custom fp32
        master-weight optimizers because their fp32 params are not registered
        in the FSDP module's parameter graph. Save optimizer/scheduler state
        directly instead.
        """
        self._save_optimizer_and_scheduler(output_dir)

    def _save_optimizer_and_scheduler(self, output_dir):
        """Bypass FSDP's optim_state_dict for our custom fp32 optimizers."""
        optimizer = self.optimizer
        if hasattr(optimizer, "optimizer"):
            # Unwrap AcceleratedOptimizer
            optimizer = optimizer.optimizer

        if isinstance(optimizer, (_FP32StateAdamW, _FP32MasterWeightOptimizer)):
            if self.args.should_save:
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(output_dir, "optimizer.pt"),
                )
                if self.lr_scheduler is not None:
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
        else:
            super()._save_optimizer_and_scheduler(output_dir)

    def _update_gamma_warmup(self):
        """Update loss_decay_gamma: gamma = gamma_init + step * epoch.

        AngelSlim DFlash gamma-warmup schedule:
            current_gamma = loss_decay_gamma + step * float(epoch)
        """
        if not self.gamma_warmup or self._gamma_init is None:
            return
        current_epoch = int(self.state.epoch) if hasattr(self.state, "epoch") else 0
        self.loss_decay_gamma = self._gamma_init + self._gamma_step * float(current_epoch)

    def prepare_data_for_draft_model(self, inputs):
        """Prepare data for DFlash training.

        Extracts hidden states from the target model. DFlash needs
        multi-layer hidden states concatenated as context features.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        loss_mask = inputs["loss_mask"]
        target_kwargs = {
            key: inputs[key]
            for key in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "input_features",
                "input_features_mask",
                "input_position_ids",
                "image_position_ids",
                "mm_token_type_ids",
                "raw_images",
            )
            if key in inputs and inputs[key] is not None
        }

        # Get hidden states from target model
        hidden_states = self.target_model.get_hidden_states_only(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
            **target_kwargs,
        )

        return {
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "token_position_ids": self._serving_position_ids(
                input_ids, attention_mask, target_kwargs
            ),
        }

    def _serving_position_ids(self, input_ids, attention_mask, target_kwargs):
        """Position ids the vLLM drafter actually reads.

        Qwen3.5 M-RoPE is ``(3, batch, seq)``: time, height, and width.
        Image tokens share a time id; height and width separate them.
        """
        mm_token_type_ids = target_kwargs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            return None
        backend = getattr(self.target_model, "backend", None)
        outer = getattr(backend, "model", None)
        inner = getattr(outer, "model", None)
        if inner is None or not hasattr(inner, "get_rope_index"):
            return None
        rope_positions, _ = inner.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=target_kwargs.get("image_grid_thw"),
            video_grid_thw=target_kwargs.get("video_grid_thw"),
            attention_mask=attention_mask,
        )
        return rope_positions.to(device=input_ids.device, dtype=torch.long)

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask).

        Always returns tensors so every rank runs the same FSDP draft forward.
        Rows with no eligible anchors keep a dummy position 0 with keep_mask
        False. A single eligible position is enough: the block still has up to
        ``block_size-1`` prediction targets. Requiring two used to skip the
        hard-tail serving anchor when the 16-token window sat at EOS.
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        if os.environ.get("ANGELSLIM_DFLASH_ANCHOR_TO_END") == "1":
            # Serving anchors inside the answer's last block_size tokens too
            # (e.g. "status: CONTINUE<|im_end|>"); labels past the end are
            # masked by valid_label_mask.
            max_anchor = max(seq_len - 2, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_valid = int(valid_counts.max().item())
        max_n = min(self.num_anchors, max(max_valid, 1))

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1, device=device))

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(
            1
        ).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device))

        return anchors, keep_mask

    def _create_position_ids(
        self,
        anchor_positions: torch.Tensor,
        block_size: int | None = None,
    ) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        bs = self.block_size if block_size is None else int(block_size)
        offsets = torch.arange(bs, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _block_logits(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        autoregressive_prefix: bool = False,
        block_size: int | None = None,
        token_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the draft over the sampled blocks and return vocabulary logits."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        bs = self.block_size if block_size is None else int(block_size)

        noise_embedding = self._create_noise_embed(
            input_ids,
            anchor_positions,
            block_keep_mask,
            autoregressive_prefix=autoregressive_prefix,
            block_size=bs,
        )
        if token_position_ids is None:
            context_position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            )
            draft_anchor_positions = anchor_positions
            draft_position_ids = self._create_position_ids(
                draft_anchor_positions,
                block_size=bs,
            )
            full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        elif token_position_ids.dim() == 3 and token_position_ids.shape[0] == 3:
            context_position_ids = token_position_ids.to(device=device, dtype=torch.long)
            anchor_index = anchor_positions.clamp(max=seq_len - 1)
            batch_index = torch.arange(bsz, device=device)[:, None]
            anchor_mrope = context_position_ids[:, batch_index, anchor_index]
            offsets = torch.arange(bs, device=device).view(1, 1, 1, -1)
            draft_position_ids = (anchor_mrope.unsqueeze(-1) + offsets).reshape(3, bsz, -1)
            full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=-1)
        else:
            context_position_ids = token_position_ids.to(device=device, dtype=torch.long)
            draft_anchor_positions = torch.gather(
                context_position_ids,
                1,
                anchor_positions.clamp(max=seq_len - 1),
            )
            draft_position_ids = self._create_position_ids(
                draft_anchor_positions,
                block_size=bs,
            )
            full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        dflash_attn_mask = create_dflash_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=seq_len,
            block_size=bs,
            device=device,
            causal_within_block=autoregressive_prefix,
            attention_backend=self.attention_backend,
        )

        model_dtype = next(model.parameters()).dtype
        output_hidden = model(
            noise_embedding=noise_embedding.to(model_dtype),
            target_hidden=hidden_states.to(model_dtype),
            attention_mask=dflash_attn_mask,
            position_ids=full_position_ids,
        )

        output_hidden = output_hidden.to(self.target_lm_head.weight.dtype)
        logits = self.target_lm_head(output_hidden)
        if self.target_logit_softcap is not None:
            logits = logits / self.target_logit_softcap
            logits = torch.tanh(logits) * self.target_logit_softcap
        return logits.view(bsz, anchor_positions.size(1), bs, -1)

    def _rejection_point_anchors(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        token_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Move each anchor to the position where its block would be rejected.

        Uniformly sampled anchors do not match serving. At serving time only a
        request's first block starts at a freely chosen anchor; every later block
        starts exactly where the previous block was rejected, which is a token
        the draft just failed to predict. Measured on the 60-prompt set, blocks
        at a free anchor commit 6.84 tokens while blocks at a rejection anchor
        commit 3.27, and rejection anchors are 95% of steps. Training on
        uniform anchors therefore optimizes a distribution the model almost
        never sees.

        One extra no-grad forward locates each block's first mismatch and
        returns the anchor serving would visit next.
        """
        seq_len = input_ids.shape[1]
        bs = self.block_size
        device = input_ids.device

        with torch.no_grad():
            logits = self._block_logits(
                model,
                input_ids,
                hidden_states,
                anchor_positions,
                block_keep_mask,
                token_position_ids=token_position_ids,
            )
            predictions = logits.argmax(dim=-1)  # [B, N, bs]

            offsets = torch.arange(bs, device=device).view(1, 1, -1)
            label_indices = (anchor_positions.unsqueeze(-1) + offsets).clamp(
                max=seq_len - 1
            )
            targets = torch.gather(
                input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                dim=2,
                index=label_indices,
            )

            mismatch = predictions != targets
            # Block position 0 carries the anchor token itself and is never a
            # prediction, so it can never be the rejection point.
            mismatch[..., 0] = False
            any_mismatch = mismatch.any(dim=-1)
            # argmax on a bool tensor returns the first True index.
            first_mismatch = mismatch.float().argmax(dim=-1)
            # A fully accepted block advances the anchor past the whole block.
            first_mismatch = torch.where(
                any_mismatch,
                first_mismatch,
                torch.full_like(first_mismatch, bs - 1),
            )
            moved = anchor_positions + first_mismatch
            max_anchor = max(seq_len - bs, 0)
            moved = moved.clamp(max=max_anchor)
        # Padding blocks keep their original anchor; their loss weight is zero.
        return torch.where(block_keep_mask, moved, anchor_positions)

    def _create_noise_embed(
        self,
        input_ids,
        anchor_positions,
        block_keep_mask,
        autoregressive_prefix: bool = False,
        block_size: int | None = None,
    ):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size if block_size is None else int(block_size)
        device = input_ids.device

        noise_ids = torch.full((bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device)

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )
        if autoregressive_prefix:
            offsets = torch.arange(1, bs, device=device).view(1, 1, -1)
            predecessor_indices = (
                anchor_positions.unsqueeze(-1) + offsets - 1
            ).clamp(max=seq_len - 1)
            predecessor_tokens = torch.gather(
                input_ids.unsqueeze(1).expand(-1, n, -1),
                dim=2,
                index=predecessor_indices,
            )
            block_offsets = (
                torch.arange(n, device=device).view(1, n, 1) * bs + offsets
            ).expand(bsz, -1, -1)
            valid = block_keep_mask.unsqueeze(-1).expand_as(predecessor_tokens)
            noise_ids.scatter_(
                1,
                block_offsets.reshape(bsz, -1),
                torch.where(
                    valid,
                    predecessor_tokens,
                    torch.full_like(predecessor_tokens, self.mask_token_id),
                ).reshape(bsz, -1),
            )

        return self.target_embed_tokens(noise_ids) * self.target_embedding_scale

    def _compute_dflash_loss_and_accuracy(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        return_position_metrics: bool = False,
        token_position_ids: torch.Tensor | None = None,
    ):
        """Core DFlash block-parallel loss logic (shared by train + eval).

        Steps:
          1. Sample anchor positions from valid loss_mask positions.
          2. Build noise embedding (anchor token is real, rest are MASK).
          3. Build DFlash BlockMask (context-causal + intra-block bidirectional).
          4. Run draft model forward → logits.
          5. Compute weighted CE loss with optional exponential decay.
          6. Compute accuracy (no-decay mask).

        Returns:
            ``(loss, accuracy)`` by default. When ``return_position_metrics``
            is true, also returns additive per-position metric counts.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # ── 1. Anchor sampling ────────────────────────────────────────────────
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        # No valid anchors → return zero loss connected to model params (DDP-safe)
        if anchor_positions is None:
            # Keep every parameter connected so FSDP sees a consistent gradient
            # structure across ranks. Reduce each local shard in FP32:
            # flattening an uneven global DTensor is unsupported, and a global
            # reduction here would deadlock when only one rank has a short row.
            zero_loss = sum(
                (p.to_local() if hasattr(p, "to_local") else p).float().sum() * 0.0
                for p in model.parameters()
                if p.requires_grad and p.numel()
            )
            result = (zero_loss, torch.tensor(0.0, device=device))
            if return_position_metrics:
                return (*result, empty_dflash_position_counts(self.block_size, device))
            return result

        # ── 2. Optionally move anchors onto the serving distribution ─────────
        if self.hard_anchor_fraction > 0.0:
            moved = self._rejection_point_anchors(
                model,
                input_ids,
                hidden_states,
                anchor_positions,
                block_keep_mask,
                token_position_ids=token_position_ids,
            )
            if self.hard_anchor_fraction >= 1.0:
                anchor_positions = moved
            else:
                take = torch.rand_like(moved, dtype=torch.float) < (
                    self.hard_anchor_fraction
                )
                anchor_positions = torch.where(take, moved, anchor_positions)

        # ── 3. Draft model forward → logits  [B, N, bs, vocab] ──────────────
        logits = self._block_logits(
            model,
            input_ids,
            hidden_states,
            anchor_positions,
            block_keep_mask,
            token_position_ids=token_position_ids,
        )

        # ── 6. Labels: position k in block predicts token at (anchor + k) ────
        bs = self.block_size
        label_offsets = torch.arange(0, bs, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            dim=2,
            index=safe_label_indices,
        )  # [B, N, bs]

        # ── 7. Weight mask: valid block × in-bounds × skip anchor × loss_mask ─
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, bs).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(bs, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()  # skip pos 0 (anchor)
        if self.max_train_positions is not None:
            weight_mask = weight_mask * (
                pos_in_block <= int(self.max_train_positions)
            ).float()

        gathered_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            dim=2,
            index=safe_label_indices,
        )
        weight_mask = weight_mask * gathered_loss_mask

        binary_eval_mask = weight_mask.view(-1)  # no decay, used for accuracy

        # ── 8. Optional block-horizon and exponential position weights ────────
        if self.horizon_boundary is not None and self.horizon_tail_weight != 1.0:
            horizon_weights = torch.where(
                pos_in_block >= int(self.horizon_boundary),
                torch.tensor(self.horizon_tail_weight, device=device),
                torch.tensor(1.0, device=device),
            )
            weight_mask = weight_mask * horizon_weights

        # Exponential decay: exp(-(k-1)/γ), k=1 gets weight 1.0.
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(bs, device=device).view(1, 1, -1)
            decay = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            weight_mask = weight_mask * decay

        # ── 9. Cross-entropy loss ─────────────────────────────────────────────
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        dynamic_weights = None
        if self.acceptance_aware_loss:
            dynamic_weights = acceptance_aware_position_weights(
                loss_per_token.view_as(target_ids),
                binary_eval_mask.view_as(target_ids),
                alpha=self.dpace_alpha,
            )
            flat_weights = flat_weights * dynamic_weights.view(-1)
        loss = (loss_per_token * flat_weights).sum() / (flat_weights.sum() + 1e-6)
        if self.ar_aux_weight > 0:
            ar_logits = self._block_logits(
                model,
                input_ids,
                hidden_states,
                anchor_positions,
                block_keep_mask,
                autoregressive_prefix=True,
                token_position_ids=token_position_ids,
            )
            ar_loss_per_token = F.cross_entropy(
                ar_logits.view(-1, ar_logits.size(-1)),
                flat_targets,
                reduction="none",
            )
            ar_loss = (ar_loss_per_token * flat_weights).sum() / (
                flat_weights.sum() + 1e-6
            )
            loss = loss + self.ar_aux_weight * ar_loss
        if self.front_distill_weight > 0:
            assert self.front_teacher_model is not None
            teacher = self.front_teacher_model
            teacher_param = next(teacher.parameters())
            model_dtype = next(model.parameters()).dtype
            if teacher_param.device != device or teacher_param.dtype != model_dtype:
                teacher.to(device=device, dtype=model_dtype)
            teacher.eval()
            with torch.no_grad():
                teacher_logits = self._block_logits(
                    teacher,
                    input_ids,
                    hidden_states,
                    anchor_positions,
                    block_keep_mask,
                    block_size=self.front_teacher_block_size,
                    token_position_ids=token_position_ids,
                )
            front_loss = front_distillation_kl(
                logits,
                teacher_logits,
                binary_eval_mask.view_as(target_ids),
                num_positions=self.front_distill_positions,
                temperature=self.front_distill_temperature,
                position_weights=dynamic_weights,
            )
            loss = loss + self.front_distill_weight * front_loss

        # ── 10. Accuracy (no gradient) ────────────────────────────────────────
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            accuracy = correct.sum().float() / (binary_eval_mask.sum() + 1e-6)

        if return_position_metrics:
            with torch.no_grad():
                position_counts = dflash_position_metric_counts(
                    predictions=pred_ids.view_as(target_ids),
                    targets=target_ids,
                    valid_mask=binary_eval_mask.view_as(target_ids),
                    token_losses=loss_per_token.view_as(target_ids),
                )
            return loss, accuracy, position_counts
        return loss, accuracy

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
        return_outputs: bool = False,
    ) -> torch.Tensor:
        """Compute the DFlash training loss.

        Unlike Eagle3's iterative multi-step loss, DFlash computes a single
        block-parallel cross-entropy loss over all sampled anchor positions.
        """
        # Update gamma if warmup is enabled (no-op when gamma_warmup=False)
        self._update_gamma_warmup()

        data = self.prepare_data_for_draft_model(inputs)

        loss, accuracy, position_counts = self._compute_dflash_loss_and_accuracy(
            model=model,
            input_ids=data["input_ids"],
            hidden_states=data["hidden_states"],
            loss_mask=data["loss_mask"],
            return_position_metrics=True,
            token_position_ids=data.get("token_position_ids"),
        )

        logged = {
            "train/loss": round(float(loss.item()), 4),
            "train/accuracy": round(float(accuracy.item()), 4),
        }
        logging_steps = max(int(getattr(self.args, "logging_steps", 10) or 10), 1)
        step = int(getattr(getattr(self, "state", None), "global_step", 0))
        if step % logging_steps == 0:
            by_position = {
                item["position"]: item
                for item in finalize_dflash_position_metrics(position_counts)
            }
            # Position 1 is the first drafted token. Position 15 is the K15
            # proxy: prefix accuracy there means the whole block was correct.
            for position in (1, 8, 15):
                item = by_position.get(position)
                if item is None:
                    continue
                top1 = item["independent_top1_accuracy"]
                prefix = item["cumulative_prefix_accuracy"]
                if top1 is not None:
                    logged[f"train/pos{position}_top1"] = round(top1, 4)
                if prefix is not None:
                    logged[f"train/pos{position}_prefix"] = round(prefix, 4)
        self.log(logged)

        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation step."""
        data = self.prepare_data_for_draft_model(inputs)

        with torch.no_grad():
            loss, accuracy, position_counts = self._compute_dflash_loss_and_accuracy(
                model=model,
                input_ids=data["input_ids"],
                hidden_states=data["hidden_states"],
                loss_mask=data["loss_mask"],
                return_position_metrics=True,
                token_position_ids=data.get("token_position_ids"),
            )

        eval_metrics = {
            "eval/loss": round(float(loss.item()), 4),
            "eval/accuracy": round(float(accuracy.item()), 4),
        }
        for metrics in finalize_dflash_position_metrics(position_counts):
            position = metrics["position"]
            for name in (
                "independent_top1_accuracy",
                "cumulative_prefix_accuracy",
                "conditional_prefix_accuracy",
                "ce_loss",
            ):
                value = metrics[name]
                if value is not None:
                    eval_metrics[f"eval/position_{position}/{name}"] = round(value, 6)
        self.log(eval_metrics)

        return loss, None, None
