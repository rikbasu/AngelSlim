"""Per-position metrics shared by DFlash/DFlare training and evaluation."""

from __future__ import annotations

from typing import Dict, Optional

import torch


_COUNT_KEYS = (
    "independent_correct",
    "independent_total",
    "prefix_correct",
    "prefix_total",
    "conditional_total",
    "ce_sum",
)


def select_deterministic_anchors(
    loss_mask: torch.Tensor,
    block_size: int,
    num_anchors: Optional[int],
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    """Select reproducible trainer-compatible anchors for one sequence.

    An anchor must itself be in the loss span and leave room for a complete
    draft block. When a limit is supplied, anchors are sampled reproducibly
    from a CPU generator and returned in sequence order.
    """
    if loss_mask.ndim != 1:
        raise ValueError(f"loss_mask must be 1-D, got shape {tuple(loss_mask.shape)}")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if num_anchors is not None and num_anchors < 1:
        raise ValueError("num_anchors must be positive or None")

    max_anchor = loss_mask.numel() - block_size
    if max_anchor < 0:
        return torch.empty(0, dtype=torch.long)

    candidates = torch.nonzero(
        loss_mask[: max_anchor + 1].detach().cpu() > 0.5,
        as_tuple=False,
    ).flatten()
    if num_anchors is None or candidates.numel() <= num_anchors:
        return candidates

    generator = torch.Generator(device="cpu")
    mixed_seed = (int(seed) + 1_000_003 * int(sample_index)) % (2**63 - 1)
    generator.manual_seed(mixed_seed)
    selected = candidates[
        torch.randperm(candidates.numel(), generator=generator)[:num_anchors]
    ]
    return selected.sort().values


def dflash_position_metric_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    token_losses: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return additive counts for positions 1..block_size-1.

    Inputs have any number of leading dimensions and a final block-position
    dimension. Position zero (the known anchor token) is intentionally
    excluded. Prefix metrics require every loss-mask position in the prefix to
    be valid, so masked gaps never count as successful speculative prefixes.
    """
    if not (
        predictions.shape == targets.shape == valid_mask.shape == token_losses.shape
    ):
        raise ValueError(
            "predictions, targets, valid_mask, and token_losses must have "
            f"identical shapes; got {tuple(predictions.shape)}, "
            f"{tuple(targets.shape)}, {tuple(valid_mask.shape)}, and "
            f"{tuple(token_losses.shape)}"
        )
    if predictions.ndim < 1 or predictions.shape[-1] < 2:
        raise ValueError("metric inputs must include anchor plus at least one position")

    correct = predictions[..., 1:].eq(targets[..., 1:])
    valid = valid_mask[..., 1:] > 0.5
    losses = token_losses[..., 1:].float()

    position_count = correct.shape[-1]
    correct = correct.reshape(-1, position_count)
    valid = valid.reshape(-1, position_count)
    losses = losses.reshape(-1, position_count)

    eligible_prefix = valid.cumprod(dim=-1).bool()
    correct_prefix = (correct & valid).cumprod(dim=-1).bool() & eligible_prefix
    previous_correct = torch.cat(
        (
            torch.ones(
                (correct_prefix.shape[0], 1),
                dtype=torch.bool,
                device=correct_prefix.device,
            ),
            correct_prefix[:, :-1],
        ),
        dim=-1,
    )

    return {
        "independent_correct": (correct & valid).sum(dim=0).to(torch.float64),
        "independent_total": valid.sum(dim=0).to(torch.float64),
        "prefix_correct": correct_prefix.sum(dim=0).to(torch.float64),
        "prefix_total": eligible_prefix.sum(dim=0).to(torch.float64),
        "conditional_total": (eligible_prefix & previous_correct)
        .sum(dim=0)
        .to(torch.float64),
        "ce_sum": (losses * valid).sum(dim=0).to(torch.float64),
    }


def empty_dflash_position_counts(
    block_size: int,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Create an accumulator for positions 1..block_size-1."""
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    return {
        key: torch.zeros(block_size - 1, dtype=torch.float64, device=device)
        for key in _COUNT_KEYS
    }


def add_dflash_position_counts(
    accumulator: Dict[str, torch.Tensor],
    counts: Dict[str, torch.Tensor],
) -> None:
    """Add one metric-count dictionary into an accumulator in place."""
    for key in _COUNT_KEYS:
        accumulator[key].add_(counts[key].to(accumulator[key].device))


def finalize_dflash_position_metrics(
    counts: Dict[str, torch.Tensor],
) -> list[dict]:
    """Convert additive counts into JSON-friendly per-position metrics."""

    def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> Optional[float]:
        denominator_value = float(denominator.item())
        if denominator_value == 0:
            return None
        return float(numerator.item()) / denominator_value

    metrics = []
    for index in range(counts["independent_total"].numel()):
        metrics.append(
            {
                "position": index + 1,
                "independent_top1_accuracy": ratio(
                    counts["independent_correct"][index],
                    counts["independent_total"][index],
                ),
                "cumulative_prefix_accuracy": ratio(
                    counts["prefix_correct"][index],
                    counts["prefix_total"][index],
                ),
                "conditional_prefix_accuracy": ratio(
                    counts["prefix_correct"][index],
                    counts["conditional_total"][index],
                ),
                "ce_loss": ratio(
                    counts["ce_sum"][index],
                    counts["independent_total"][index],
                ),
                "token_count": int(counts["independent_total"][index].item()),
                "prefix_count": int(counts["prefix_total"][index].item()),
                "conditional_count": int(counts["conditional_total"][index].item()),
            }
        )
    return metrics
