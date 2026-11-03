"""Gemma 4 DFlare draft model.

This module intentionally keeps the draft network independent from the
Gemma-4 target implementation.  The target supplies fused hidden states and
its embedding/lm-head weights; the draft only needs the Gemma text dimensions
and attention geometry.  That makes the same checkpoint usable with the
Transformers and vLLM target wrappers once both expose the same hidden-state
contract.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from .draft_model_factory import DraftModelFactory

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # pragma: no cover - depends on the installed torch build.
    create_block_mask = None
    flex_attention = None

_compiled_flex_attention = None


def _get_compiled_flex_attention():
    global _compiled_flex_attention
    if flex_attention is None:
        return None
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
    return _compiled_flex_attention


class Gemma4DFlareConfig(PretrainedConfig):
    """Configuration for a Gemma-4-conditioned DFlare draft network."""

    model_type = "gemma4_dflare"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 2560,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 7,
        num_target_layers: int = 9,
        target_hidden_size: int = 2560,
        target_embedding_size: int | None = None,
        target_head_hidden_size: int | None = None,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        block_size: int = 16,
        num_anchors: int = 512,
        mask_token_id: int = 4,
        target_layer_ids: Optional[List[int]] = None,
        attention_backend: str = "eager",
        tail_adapter_rank: int = 0,
        tail_adapter_start: int = 16,
        tail_adapter_scale: float = 1.0,
        dflare_config: Optional[dict] = None,
        dflash_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_target_layers = num_target_layers
        self.target_hidden_size = target_hidden_size
        self.target_embedding_size = int(
            target_embedding_size or target_hidden_size
        )
        self.target_head_hidden_size = int(
            target_head_hidden_size or self.target_embedding_size
        )
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.tail_adapter_rank = int(tail_adapter_rank)
        self.tail_adapter_start = int(tail_adapter_start)
        self.tail_adapter_scale = float(tail_adapter_scale)
        self.dflare_config = dict(dflare_config or dflash_config or {})
        # Keep the legacy key populated because the shared AngelSlim DFlash
        # trainer reads it for target-layer capture and mask-token settings.
        self.dflash_config = self.dflare_config
        if target_layer_ids is not None:
            self.dflare_config["target_layer_ids"] = list(target_layer_ids)
        self.dflare_config.setdefault(
            "target_layer_ids",
            list(range(self.num_target_layers)),
        )
        self.dflare_config.setdefault("mask_token_id", self.mask_token_id)
        self.num_target_layers = len(self.dflare_config["target_layer_ids"])
        self._attn_implementation = attention_backend
        self.torch_dtype = kwargs.get("torch_dtype", "bfloat16")


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(hidden_states.dtype) * self.weight


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
    layout: str = "legacy",
) -> torch.Tensor:
    """Apply standard rotary embeddings to [B, H, S, D] states."""
    head_dim = hidden_states.shape[-1]
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=hidden_states.device).float()
            / head_dim
        )
    )
    frequencies = torch.einsum(
        "bs,d->bsd",
        position_ids.float(),
        inv_freq,
    )
    if layout == "neox":
        # _rotate_half pairs the first and second halves, so both halves use
        # the same frequencies.
        cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1).unsqueeze(1)
        sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1).unsqueeze(1)
    elif layout == "legacy":
        # Compatibility for checkpoints trained before the NeoX pairing fix.
        cos = frequencies.cos().repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = frequencies.sin().repeat_interleave(2, dim=-1).unsqueeze(1)
    else:
        raise ValueError(f"Unsupported DFlare RoPE layout: {layout}")
    return hidden_states * cos.to(hidden_states.dtype) + _rotate_half(
        hidden_states
    ) * sin.to(hidden_states.dtype)


class _Gemma4DFlareAttention(nn.Module):
    def __init__(self, config: Gemma4DFlareConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.rope_layout = config.dflare_config.get("rope_layout", "legacy")
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.k_proj_target = nn.Linear(
            config.target_hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj_target = nn.Linear(
            config.target_hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        context_position_ids: torch.Tensor,
        draft_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, draft_length, _ = hidden_states.shape
        context_length = target_hidden.shape[1]

        query = self.q_proj(hidden_states).view(
            batch_size, draft_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        context_key = self.k_proj_target(target_hidden).view(
            batch_size, context_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        draft_key = self.k_proj(hidden_states).view(
            batch_size, draft_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        context_value = self.v_proj_target(target_hidden).view(
            batch_size, context_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        draft_value = self.v_proj(hidden_states).view(
            batch_size, draft_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        query = self.q_norm(query)
        context_key = self.k_norm(context_key)
        draft_key = self.k_norm(draft_key)
        query = _apply_rope(
            query, draft_position_ids, self.rope_theta, self.rope_layout
        )
        context_key = _apply_rope(
            context_key,
            context_position_ids,
            self.rope_theta,
            self.rope_layout,
        )
        draft_key = _apply_rope(
            draft_key,
            draft_position_ids,
            self.rope_theta,
            self.rope_layout,
        )

        key = torch.cat((context_key, draft_key), dim=2)
        value = torch.cat((context_value, draft_value), dim=2)
        key = key.repeat_interleave(self.num_kv_groups, dim=1)
        value = value.repeat_interleave(self.num_kv_groups, dim=1)

        if (
            self._uses_flex_attention(attention_mask)
            and create_block_mask is not None
            and flex_attention is not None
        ):
            compiled_flex_attention = _get_compiled_flex_attention()
            assert compiled_flex_attention is not None
            attended = compiled_flex_attention(
                query,
                key,
                value,
                block_mask=attention_mask,
            )
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        return self.o_proj(attended.transpose(1, 2).reshape(
            batch_size,
            draft_length,
            self.num_heads * self.head_dim,
        ))

    @staticmethod
    def _uses_flex_attention(attention_mask: Optional[torch.Tensor]) -> bool:
        return attention_mask is not None and attention_mask.__class__.__name__ == "BlockMask"


class _Gemma4DFlareLayer(nn.Module):
    def __init__(self, config: Gemma4DFlareConfig):
        super().__init__()
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = _Gemma4DFlareAttention(config)
        self.post_attention_layernorm = _RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        context_position_ids: torch.Tensor,
        draft_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        hidden_states = residual + self.attention(
            normalized,
            target_hidden,
            context_position_ids,
            draft_position_ids,
            attention_mask,
        )
        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        normalized = F.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return residual + self.down_proj(normalized)


@DraftModelFactory.register
class Gemma4DFlareDraftModel(PreTrainedModel):
    """Layer-wise-fusion DFlare draft model for Gemma-4."""

    config_class = Gemma4DFlareConfig
    _supports_flex_attn = True
    _no_split_modules = ["_Gemma4DFlareLayer"]

    def __init__(self, config: Gemma4DFlareConfig):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        self.layers = nn.ModuleList(
            [_Gemma4DFlareLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_norm = _RMSNorm(
            config.target_hidden_size,
            config.rms_norm_eps,
        )
        self.input_projection = (
            nn.Linear(
                config.target_embedding_size,
                config.hidden_size,
                bias=False,
            )
            if config.target_embedding_size != config.hidden_size
            else None
        )
        self.output_projection = (
            nn.Linear(
                config.hidden_size,
                config.target_head_hidden_size,
                bias=False,
            )
            if config.target_head_hidden_size != config.hidden_size
            else None
        )
        self.layer_fusion_weights = nn.Parameter(
            torch.zeros(config.num_hidden_layers, config.num_target_layers)
        )
        for draft_layer in range(config.num_hidden_layers):
            target_layer = min(
                config.num_target_layers - 1,
                round(
                    draft_layer
                    * config.num_target_layers
                    / config.num_hidden_layers
                ),
            )
            self.layer_fusion_weights.data[draft_layer, target_layer] = 2.0
        self.block_size = config.block_size
        self.mask_token_id = config.dflare_config["mask_token_id"]
        if config.tail_adapter_rank > 0:
            self.tail_adapter_down = nn.Linear(
                config.hidden_size,
                config.tail_adapter_rank,
                bias=False,
            )
            self.tail_adapter_up = nn.Linear(
                config.tail_adapter_rank,
                config.hidden_size,
                bias=False,
            )
            nn.init.normal_(self.tail_adapter_down.weight, std=0.02)
            nn.init.zeros_(self.tail_adapter_up.weight)
        else:
            self.tail_adapter_down = None
            self.tail_adapter_up = None
        self.post_init()
        if self.tail_adapter_up is not None:
            nn.init.zeros_(self.tail_adapter_up.weight)

    def get_input_embeddings(self):
        """DFlare receives target/noise embeddings from the trainer."""
        return None

    def _make_flex_block_mask(
        self,
        anchor_positions: torch.Tensor,
        context_length: int,
        draft_length: int,
        device: torch.device,
    ):
        if create_block_mask is None:
            return None

        block_size = self.block_size

        def mask_mod(batch, _head, query_index, key_index):
            query_block = query_index // block_size
            anchor = anchor_positions[batch, query_block]
            context_key = key_index < context_length
            context_allowed = context_key & (key_index < anchor)
            draft_key = key_index >= context_length
            draft_block = (key_index - context_length) // block_size
            draft_allowed = draft_key & (draft_block == query_block)
            return context_allowed | draft_allowed

        return create_block_mask(
            mask_mod,
            B=anchor_positions.shape[0],
            H=None,
            Q_LEN=draft_length,
            KV_LEN=context_length + draft_length,
            device=device,
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if noise_embedding is None or target_hidden is None:
            raise ValueError("noise_embedding and target_hidden are required")
        batch_size, draft_length, _ = noise_embedding.shape
        context_length = target_hidden.shape[1]
        expected_width = self.config.num_target_layers * self.config.target_hidden_size
        if target_hidden.shape[-1] != expected_width:
            raise ValueError(
                "target_hidden width mismatch: "
                f"expected {expected_width}, got {target_hidden.shape[-1]}"
            )
        if draft_length % self.block_size:
            raise ValueError(
                f"draft length {draft_length} is not divisible by block_size "
                f"{self.block_size}"
            )

        target_hidden = target_hidden.view(
            batch_size,
            context_length,
            self.config.num_target_layers,
            self.config.target_hidden_size,
        )
        fusion = torch.softmax(self.layer_fusion_weights, dim=-1)
        fused = torch.einsum("bsth,dt->bsdh", target_hidden, fusion)
        fused = self.hidden_norm(fused)

        if position_ids.shape[1] >= context_length + draft_length:
            context_position_ids = position_ids[:, :context_length]
            draft_position_ids = position_ids[:, -draft_length:]
        else:
            context_position_ids = torch.arange(
                context_length,
                device=noise_embedding.device,
            ).unsqueeze(0).expand(batch_size, -1)
            draft_position_ids = position_ids[:, -draft_length:]

        effective_mask = attention_mask
        if (
            self.config._attn_implementation == "flex_attention"
            and noise_embedding.device.type != "cpu"
            and not self._is_block_mask(effective_mask)
        ):
            anchors = draft_position_ids[:, :: self.block_size]
            effective_mask = self._make_flex_block_mask(
                anchors,
                context_length,
                draft_length,
                noise_embedding.device,
            )
        elif effective_mask is not None and self._is_block_mask(effective_mask):
            # The trainer's BlockMask has Q=KV=S+draft_length because it was
            # built for the concatenated DFlash representation.  Rebuild it
            # for this implementation's [draft queries, context+draft KV]
            # contract.
            if noise_embedding.device.type != "cpu":
                anchors = draft_position_ids[:, :: self.block_size]
                effective_mask = self._make_flex_block_mask(
                    anchors,
                    context_length,
                    draft_length,
                    noise_embedding.device,
                )
            else:
                effective_mask = None

        hidden_states = (
            self.input_projection(noise_embedding)
            if self.input_projection is not None
            else noise_embedding
        )
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                fused[:, :, layer_index, :],
                context_position_ids,
                draft_position_ids,
                effective_mask,
            )
        hidden_states = self.norm(hidden_states)
        if self.tail_adapter_down is not None and self.tail_adapter_up is not None:
            local_offsets = torch.arange(
                draft_length,
                device=hidden_states.device,
            ) % self.block_size
            tail_mask = (
                local_offsets >= self.config.tail_adapter_start
            ).to(hidden_states.dtype).view(1, -1, 1)
            tail_residual = self.tail_adapter_up(
                F.silu(self.tail_adapter_down(hidden_states))
            )
            hidden_states = (
                hidden_states
                + self.config.tail_adapter_scale * tail_mask * tail_residual
            )
        if self.output_projection is not None:
            hidden_states = self.output_projection(hidden_states)
        return hidden_states

    @staticmethod
    def _is_block_mask(mask: Optional[torch.Tensor]) -> bool:
        return mask is not None and mask.__class__.__name__ == "BlockMask"


def extract_gemma4_context_features(
    hidden_states: tuple[torch.Tensor, ...],
    layer_ids: List[int],
) -> torch.Tensor:
    """Extract Gemma hidden states using target-layer IDs, excluding embeddings."""
    return torch.cat(
        [hidden_states[layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )
