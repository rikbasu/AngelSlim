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
from angelslim.compressor.speculative.train.models.draft.gemma4_dflare import (
    _apply_rope,
)


def _tiny_config(**overrides) -> Gemma4DFlareConfig:
    values = dict(
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
    values.update(overrides)
    return Gemma4DFlareConfig(**values)


def test_gemma4_dflare_rope_matches_neox_and_preserves_norm():
    torch.manual_seed(0)
    hidden = torch.randn(2, 4, 7, 16)
    positions = torch.arange(7).unsqueeze(0).expand(2, -1)
    theta = 10000.0
    frequencies = torch.einsum(
        "bs,d->bsd",
        positions.float(),
        1.0 / (theta ** (torch.arange(0, 16, 2).float() / 16)),
    )
    cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1).unsqueeze(1)
    sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1).unsqueeze(1)
    first, second = hidden.chunk(2, dim=-1)
    expected = hidden * cos + torch.cat((-second, first), dim=-1) * sin

    actual = _apply_rope(hidden, positions, theta, "neox")

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual.float().norm(dim=-1),
        hidden.float().norm(dim=-1),
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


def test_gemma4_dflare_projects_cross_width_target_embeddings_and_head():
    model = Gemma4DFlareDraftModel(
        _tiny_config(
            target_hidden_size=48,
            target_embedding_size=48,
            target_head_hidden_size=48,
        )
    )
    target_hidden = torch.randn(1, 6, 3 * 48)
    noise_embedding = torch.randn(1, 8, 48)
    position_ids = torch.arange(14).unsqueeze(0)

    output = model(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
    )

    assert output.shape == (1, 8, 48)
    assert model.input_projection is not None
    assert model.output_projection is not None
    assert model.layers[0].attention.k_proj_target.in_features == 48


def test_tail_adapter_is_zero_initialized_and_position_gated():
    torch.manual_seed(0)
    base = Gemma4DFlareDraftModel(_tiny_config())
    adapted = Gemma4DFlareDraftModel(
        _tiny_config(
            tail_adapter_rank=8,
            tail_adapter_start=2,
        )
    )
    result = adapted.load_state_dict(base.state_dict(), strict=False)
    assert set(result.missing_keys) == {
        "tail_adapter_down.weight",
        "tail_adapter_up.weight",
    }
    target_hidden = torch.randn(1, 6, 3 * 32)
    noise_embedding = torch.randn(1, 8, 32)
    position_ids = torch.arange(14).unsqueeze(0)

    base_output = base(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
    )
    initial_output = adapted(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
    )
    torch.testing.assert_close(initial_output, base_output)

    with torch.no_grad():
        adapted.tail_adapter_down.weight.fill_(0.1)
        adapted.tail_adapter_up.weight.fill_(0.1)
    trained_output = adapted(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
    ).view(1, 2, 4, 32)
    base_output = base_output.view(1, 2, 4, 32)

    torch.testing.assert_close(trained_output[:, :, :2], base_output[:, :, :2])
    assert not torch.allclose(trained_output[:, :, 2:], base_output[:, :, 2:])


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
    assert config.dflare_config["rope_layout"] == "legacy"
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


def test_gemma4_dflare_neox_config_is_explicit_opt_in():
    config_path = (
        Path(__file__).parents[1] / "configs" / "gemma4_dflare_neox.json"
    )
    config = DraftModelConfig.from_file(config_path)

    assert isinstance(config, Gemma4DFlareConfig)
    assert config.dflare_config["rope_layout"] == "neox"


def test_gemma4_dflare_l3t3_config_matches_slim_contract():
    config_path = (
        Path(__file__).parents[1] / "configs" / "gemma4_dflare_l3t3.json"
    )
    config = DraftModelConfig.from_file(config_path)

    assert isinstance(config, Gemma4DFlareConfig)
    assert config.num_hidden_layers == 3
    assert config.num_target_layers == 3
    assert config.dflare_config["target_layer_ids"] == [1, 21, 39]
    model = Gemma4DFlareDraftModel(config)
    assert len(model.layers) == 3
    assert model.layer_fusion_weights.shape == (3, 3)


def test_gemma4_dflare_l4t5_fallback_config():
    config_path = (
        Path(__file__).parents[1] / "configs" / "gemma4_dflare_l4t5.json"
    )
    config = DraftModelConfig.from_file(config_path)

    assert config.num_hidden_layers == 4
    assert config.num_target_layers == 5
    assert config.dflare_config["target_layer_ids"] == [1, 11, 21, 31, 39]
