import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "profile_dflare_vocab.py"
SPEC = importlib.util.spec_from_file_location("profile_dflare_vocab", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
profile_dflare_vocab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_dflare_vocab)


def test_validate_mapping_accepts_angelslim_offset_contract():
    report = profile_dflare_vocab.validate_mapping(
        [0, 1, 2, 4],
        8,
        expected_draft_vocab_size=4,
        t2d_mask=[True, False, True, False, True, False, False, True],
        required_target_ids=[0, 4],
    )

    assert report["valid"]
    assert report["target_ids"] == [0, 2, 4, 7]
    assert report["format"].startswith("offset")


def test_validate_mapping_rejects_duplicate_and_missing_special_ids():
    report = profile_dflare_vocab.validate_mapping(
        [1, 0],
        4,
        expected_draft_vocab_size=2,
        required_target_ids=[3],
    )

    assert not report["valid"]
    assert "mapped target IDs are not unique" in report["errors"]
    assert "required target IDs are absent: [3]" in report["errors"]


def test_profile_head_sizes_reports_memory_and_column_savings():
    [report] = profile_dflare_vocab.profile_head_sizes(
        target_vocab_size=100,
        hidden_size=10,
        candidate_vocab_sizes=[20],
        dtype_bytes=2,
    )

    assert report["head_bytes"] == 400
    assert report["size_reduction_percent"] == 80.0
    assert report["logit_columns_reduction_percent"] == 80.0


def test_current_vllm_audit_accepts_mapped_gemma_and_recognizes_fp8():
    vllm_source = Path(__file__).parents[2] / "vllm-nightly-dflare"
    report = profile_dflare_vocab.audit_vllm_source(vllm_source, "fp8")

    assert report["source_available"]
    assert report["heterogeneous_vocab_is_draft_model_only"]
    assert report["qwen_dflash_has_d2t_load_and_scatter"]
    assert report["gemma_dflare_mapping_supported"]
    assert not report["gemma_dflare_mapping_disabled"]
    assert report["safe_for_mapped_gemma4_dflare"]
    assert report["quantization"]["recognized"]
    assert report["quantization"]["draft_layers_use_draft_quant_config"]
    assert not report["quantization"]["gemma_dflare_lm_head_quantized"]


def test_build_report_validates_mapping_and_runtime(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "d2t": [0, 1, 2, 4],
                "t2d": [True, False, True, False, True, False, False, True],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "target": {
            "vocab_size": 8,
            "hidden_size": 4,
            "tie_word_embeddings": True,
            "bos_token_id": 0,
        },
        "draft": {
            "vocab_size": 8,
            "draft_vocab_size": 4,
            "dflare_config": {"mask_token_id": 4},
        },
        "mapping": "mapping.json",
        "candidate_vocab_sizes": [4],
        "vllm_source": str(Path(__file__).parents[2] / "vllm-nightly-dflare"),
    }

    report = profile_dflare_vocab.build_report(config, tmp_path)

    assert report["mapping"]["valid"]
    assert report["model"]["draft_input_vocab_size"] == 8
    assert report["model"]["draft_output_vocab_size"] == 4
    assert report["decision"] == "ready"
    assert report["findings"] == []


def test_example_config_still_requires_a_mapping_artifact():
    config_path = (
        Path(__file__).parents[1] / "configs" / "gemma4_dflare_vocab_audit.json"
    )

    assert profile_dflare_vocab.main(
        ["--config", str(config_path), "--require-safe"]
    ) == 2
