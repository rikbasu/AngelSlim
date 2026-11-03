import torch

from angelslim.compressor.speculative.train.trainer.dflash_position_metrics import (
    add_dflash_position_counts,
    dflash_position_metric_counts,
    empty_dflash_position_counts,
    finalize_dflash_position_metrics,
    select_deterministic_anchors,
)
from tools.evaluate_gemma4_dflare_positions import _dense_dflash_attention_mask


def test_per_position_metrics_distinguish_independent_and_prefix_accuracy():
    targets = torch.zeros((2, 4), dtype=torch.long)
    predictions = torch.tensor(
        [
            [9, 0, 0, 1],
            [9, 0, 1, 0],
        ]
    )
    valid_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ]
    )
    token_losses = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 2.0, 4.0, 6.0],
        ]
    )

    counts = empty_dflash_position_counts(block_size=4)
    add_dflash_position_counts(
        counts,
        dflash_position_metric_counts(
            predictions,
            targets,
            valid_mask,
            token_losses,
        ),
    )
    positions = finalize_dflash_position_metrics(counts)

    assert [item["independent_top1_accuracy"] for item in positions] == [
        1.0,
        0.5,
        0.5,
    ]
    assert [item["cumulative_prefix_accuracy"] for item in positions] == [
        1.0,
        0.5,
        0.0,
    ]
    assert [item["conditional_prefix_accuracy"] for item in positions] == [
        1.0,
        0.5,
        0.0,
    ]
    assert [item["ce_loss"] for item in positions] == [1.5, 3.0, 4.5]


def test_prefix_metrics_exclude_masked_gaps_but_independent_metrics_do_not():
    targets = torch.zeros((2, 4), dtype=torch.long)
    predictions = torch.zeros_like(targets)
    valid_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 0, 1],
        ]
    )
    token_losses = torch.ones((2, 4))

    positions = finalize_dflash_position_metrics(
        dflash_position_metric_counts(
            predictions,
            targets,
            valid_mask,
            token_losses,
        )
    )

    assert positions[2]["independent_top1_accuracy"] == 1.0
    assert positions[2]["token_count"] == 2
    assert positions[2]["cumulative_prefix_accuracy"] == 1.0
    assert positions[2]["prefix_count"] == 1


def test_deterministic_anchor_selection_is_reproducible_and_bounded():
    loss_mask = torch.ones(10)

    first = select_deterministic_anchors(loss_mask, 4, 3, seed=17, sample_index=2)
    second = select_deterministic_anchors(loss_mask, 4, 3, seed=17, sample_index=2)
    all_anchors = select_deterministic_anchors(
        loss_mask,
        4,
        None,
        seed=999,
        sample_index=999,
    )

    assert torch.equal(first, second)
    assert first.tolist() == sorted(first.tolist())
    assert first.numel() == 3
    assert first.min() >= 0
    assert first.max() <= 6
    assert torch.equal(all_anchors, torch.arange(7))


def test_dense_attention_mask_matches_dflash_context_and_block_contract():
    anchors = torch.tensor([2, 4])
    mask = _dense_dflash_attention_mask(
        anchors=anchors,
        context_length=6,
        block_size=3,
        device=torch.device("cpu"),
    )

    assert mask.shape == (1, 1, 6, 12)
    # First block sees context positions < 2 and only its own draft block.
    assert mask[0, 0, 0, :6].tolist() == [True, True, False, False, False, False]
    assert mask[0, 0, 0, 6:].tolist() == [True, True, True, False, False, False]
    # Second block sees context positions < 4 and only its own draft block.
    assert mask[0, 0, 3, :6].tolist() == [True, True, True, True, False, False]
    assert mask[0, 0, 3, 6:].tolist() == [False, False, False, True, True, True]
