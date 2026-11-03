import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
import torch

from angelslim.compressor.speculative.train.models.draft import (
    Gemma4DFlareConfig,
    Gemma4DFlareDraftModel,
)


REPO_ROOT = Path(__file__).parents[1]
TRAIN_TOOL_PATH = REPO_ROOT / "tools" / "train_dflash_offline.py"
RUN_SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "speculative" / "run_gemma4_dflare_offline_mi100.sh"
)


def _load_train_tool():
    spec = importlib.util.spec_from_file_location(
        "train_dflash_offline_for_test",
        TRAIN_TOOL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "target_hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_target_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "block_size": 4,
        "mask_token_id": 4,
        "target_layer_ids": [1, 2, 3],
        "attention_backend": "eager",
        "torch_dtype": "float32",
        "architectures": ["Gemma4DFlareDraftModel"],
    }
    values.update(overrides)
    return Gemma4DFlareConfig(**values)


def test_init_draft_checkpoint_loads_only_model_weights(tmp_path):
    train_tool = _load_train_tool()
    checkpoint = tmp_path / "checkpoint"
    source_model = Gemma4DFlareDraftModel(_tiny_config())
    with torch.no_grad():
        source_model.norm.weight.fill_(2.5)
    source_model.save_pretrained(checkpoint)
    # A Trainer checkpoint can contain arbitrary non-model state. The warm-start
    # path must not inspect or restore it.
    (checkpoint / "optimizer.pt").write_bytes(b"not a torch checkpoint")

    loaded_model = train_tool._load_initialized_draft_model(
        _tiny_config(),
        checkpoint,
    )

    assert torch.equal(loaded_model.norm.weight, source_model.norm.weight)


def test_init_draft_checkpoint_rejects_structural_mismatch(tmp_path):
    train_tool = _load_train_tool()
    checkpoint = tmp_path / "checkpoint"
    Gemma4DFlareDraftModel(_tiny_config()).save_pretrained(checkpoint)

    with pytest.raises(ValueError, match="hidden_size"):
        train_tool._validate_init_draft_checkpoint(
            checkpoint,
            _tiny_config(hidden_size=24),
        )


def _write_fake_python(path, captured_args):
    path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${1:-} == -c ]]; then exit 0; fi\n"
        f"printf '%s\\n' \"$@\" > {captured_args!s}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {},
            {
                "--block_size": "16",
                "--num_anchors": "32",
                "--learning_rate": "1e-4",
                "--loss_decay_gamma": "7",
            },
        ),
        (
            {
                "BLOCK_SIZE": "8",
                "NUM_ANCHORS": "12",
                "LR": "3e-5",
                "LOSS_DECAY_GAMMA": "4.5",
            },
            {
                "--block_size": "8",
                "--num_anchors": "12",
                "--learning_rate": "3e-5",
                "--loss_decay_gamma": "4.5",
            },
        ),
    ],
)
def test_gemma4_offline_script_honors_defaults_and_overrides(
    tmp_path,
    overrides,
    expected,
):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    job_dir = tmp_path / "job"
    cache_dir.mkdir()
    captured_args = tmp_path / "captured-args"
    fake_python = tmp_path / "python"
    _write_fake_python(fake_python, captured_args)

    env = {
        **os.environ,
        "PYTHON": str(fake_python),
        "JOB_DIR": str(job_dir),
        **overrides,
    }
    subprocess.run(
        ["bash", str(RUN_SCRIPT_PATH), str(cache_dir), str(output_dir), "2"],
        check=True,
        env=env,
    )

    arguments = captured_args.read_text(encoding="utf-8").splitlines()
    for flag, value in expected.items():
        assert arguments[arguments.index(flag) + 1] == value
    assert (job_dir / "TRAINING_DONE").is_file()


def test_gemma4_offline_script_passes_init_checkpoint_and_is_reboot_safe(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    init_checkpoint = tmp_path / "init-checkpoint"
    job_dir = tmp_path / "job"
    cache_dir.mkdir()
    init_checkpoint.mkdir()
    captured_args = tmp_path / "captured-args"
    fake_python = tmp_path / "python"
    _write_fake_python(fake_python, captured_args)
    env = {
        **os.environ,
        "PYTHON": str(fake_python),
        "JOB_DIR": str(job_dir),
        "INIT_DRAFT_CHECKPOINT": str(init_checkpoint),
    }

    command = ["bash", str(RUN_SCRIPT_PATH), str(cache_dir), str(output_dir)]
    subprocess.run(command, check=True, env=env)
    arguments = captured_args.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--init_draft_checkpoint") + 1] == str(
        init_checkpoint
    )

    captured_args.unlink()
    subprocess.run(command, check=True, env=env)
    assert not captured_args.exists()
