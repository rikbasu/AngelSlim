import pytest
import torch

from angelslim.compressor.speculative.train.trainer.online_dflash_trainer import (
    acceptance_aware_position_weights,
    freeze_front_teacher_model,
    front_distillation_kl,
)


def _normalized_expected_weights(
    probabilities: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    smoothed = (1.0 - alpha) * probabilities + alpha
    prefix = torch.cumprod(smoothed, dim=-1)
    weights = torch.flip(
        torch.cumsum(torch.flip(prefix, dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    return weights / weights.mean()


def test_dpace_weights_use_asymmetric_confidence_smoothing():
    probabilities = torch.tensor([[0.2, 0.4, 0.8]])
    token_nll = -probabilities.log()
    valid_mask = torch.ones_like(token_nll)

    actual = acceptance_aware_position_weights(
        token_nll,
        valid_mask,
        alpha=0.5,
    )

    expected = _normalized_expected_weights(probabilities, alpha=0.5)
    torch.testing.assert_close(actual, expected)


def test_dpace_smoothing_preserves_later_position_signal():
    token_nll = -torch.tensor([[1.0e-6, 0.5, 0.5]]).log()
    valid_mask = torch.ones_like(token_nll)

    unsmoothed = acceptance_aware_position_weights(
        token_nll,
        valid_mask,
        alpha=0.0,
    )
    smoothed = acceptance_aware_position_weights(
        token_nll,
        valid_mask,
        alpha=0.5,
    )

    assert smoothed[0, -1] > unsmoothed[0, -1] * 1.5
    torch.testing.assert_close(smoothed.mean(), torch.tensor(1.0))


def test_dpace_weights_are_detached_and_mask_invalid_positions():
    token_nll = torch.tensor([[0.2, 0.4, 0.8]], requires_grad=True)
    valid_mask = torch.tensor([[1.0, 1.0, 0.0]])

    weights = acceptance_aware_position_weights(token_nll, valid_mask)

    assert not weights.requires_grad
    assert weights[0, -1] == 0.0
    torch.testing.assert_close(weights[0, :2].mean(), torch.tensor(1.0))


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_dpace_rejects_invalid_smoothing(alpha):
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        acceptance_aware_position_weights(
            torch.zeros(1, 2),
            torch.ones(1, 2),
            alpha=alpha,
        )


def test_front_distillation_ignores_anchor_and_tail_positions():
    teacher = torch.zeros(1, 1, 24, 5)
    student = teacher.clone()
    student[..., 0, 0] = 10.0
    student[..., 16:, 0] = 10.0
    student.requires_grad_(True)
    valid = torch.ones(1, 1, 24)

    loss = front_distillation_kl(
        student,
        teacher,
        valid,
        num_positions=15,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_front_distillation_backpropagates_only_through_front():
    teacher = torch.zeros(1, 1, 24, 5)
    student = torch.zeros_like(teacher, requires_grad=True)
    with torch.no_grad():
        teacher[..., 4, 2] = 3.0
    valid = torch.ones(1, 1, 24)

    loss = front_distillation_kl(
        student,
        teacher,
        valid,
        num_positions=15,
        temperature=1.0,
    )
    loss.backward()

    assert loss > 0
    assert student.grad[..., 1:16, :].abs().sum() > 0
    assert student.grad[..., 0, :].abs().sum() == 0
    assert student.grad[..., 16:, :].abs().sum() == 0


def test_freeze_front_teacher_model_disables_gradients():
    teacher = torch.nn.Linear(4, 4)

    returned = freeze_front_teacher_model(teacher)

    assert returned is teacher
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
