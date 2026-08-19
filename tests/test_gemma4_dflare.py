import json
from pathlib import Path

import torch

from angelslim.compressor.speculative.train.models.draft import (
    Gemma4DFlareConfig,
    Gemma4DFlareDraftModel,
)
from angelslim.compressor.speculative.train.models.draft.draft_model_factory import (
    DraftModelConfig,
)


def _tiny_config() -> Gemma4DFlareConfig:
    return Gemma4DFlareConfig(
        vocab_size=128,
        hidden_size=32,
        target_hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_target_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        block_size=4,
        mask_token_id=4,
        target_layer_ids=[1, 3, 5],
        attention_backend="eager",
        torch_dtype="float32",
    )


def test_gemma4_dflare_forward_shape_and_target_width():
    torch.manual_seed(0)
    model = Gemma4DFlareDraftModel(_tiny_config())
    target_hidden = torch.randn(2, 6, 3 * 32)
    noise_embedding = torch.randn(2, 8, 32)
    position_ids = torch.arange(14).unsqueeze(0).expand(2, -1)

    output = model(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
    )

    assert output.shape == (2, 8, 32)
    assert torch.isfinite(output).all()


def test_gemma4_dflare_rejects_incompatible_target_width():
    model = Gemma4DFlareDraftModel(_tiny_config())
    with torch.no_grad():
        target_hidden = torch.randn(1, 6, 2 * 32)
        noise_embedding = torch.randn(1, 8, 32)
        position_ids = torch.arange(14).unsqueeze(0)

    try:
        model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
        )
    except ValueError as exc:
        assert "target_hidden width mismatch" in str(exc)
    else:
        raise AssertionError("incompatible target width was accepted")


def test_gemma4_dflare_supports_training_gradients():
    torch.manual_seed(0)
    model = Gemma4DFlareDraftModel(_tiny_config())
    target_head = torch.nn.Linear(32, 128, bias=False)
    target_hidden = torch.randn(2, 6, 3 * 32)
    noise_embedding = torch.randn(2, 8, 32)
    position_ids = torch.arange(14).unsqueeze(0).expand(2, -1)
    labels = torch.randint(0, 128, (2, 8))

    logits = target_head(
        model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
        )
    )
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_gemma4_dflare_config_file_registers_model():
    config_path = (
        Path(__file__).parents[1] / "configs" / "gemma4_dflare.json"
    )
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_data["architectures"] == ["Gemma4DFlareDraftModel"]
    config = DraftModelConfig.from_file(config_path)
    assert isinstance(config, Gemma4DFlareConfig)
    assert config.dflare_config["mask_token_id"] == 4
    assert config.dflare_config["target_layer_ids"] == [
        1,
        6,
        11,
        16,
        21,
        26,
        31,
        36,
        39,
    ]
