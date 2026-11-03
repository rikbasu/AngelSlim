#!/usr/bin/env python3
"""Profile reduced-vocabulary DFlare heads without creating unsafe artifacts.

The utility is intentionally read-only.  It validates the EAGLE ``d2t`` mapping
contract, estimates output-head savings, audits the checked-out vLLM runtime,
and reports whether a mapped Gemma-4 DFlare checkpoint can be consumed safely.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


SPECIAL_ID_KEYS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
    "mask_token_id",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_mapping(path: Path) -> tuple[list[int], list[bool] | None]:
    """Load AngelSlim's d2t-offset/t2d-mask artifact."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        state = _read_json(path)
    elif suffix in {".pt", ".pth", ".bin"}:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - normal install has torch.
            raise RuntimeError("PyTorch is required to inspect .pt mappings") from exc
        state = torch.load(path, map_location="cpu", weights_only=True)
    elif suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - declared core dependency.
            raise RuntimeError("safetensors is required for this mapping") from exc
        state = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.rsplit(".", 1)[-1] in {"d2t", "t2d"}:
                    state[name.rsplit(".", 1)[-1]] = handle.get_tensor(name)
    else:
        raise ValueError(f"Unsupported mapping format: {path.suffix}")

    if not isinstance(state, dict) or "d2t" not in state:
        raise ValueError(f"{path} does not contain a d2t mapping")

    def as_list(value: Any) -> list[Any]:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            raise ValueError("mapping tensors must be one-dimensional")
        return value

    d2t = [int(value) for value in as_list(state["d2t"])]
    t2d = None
    if "t2d" in state:
        t2d = [bool(value) for value in as_list(state["t2d"])]
    return d2t, t2d


def validate_mapping(
    d2t_offsets: list[int],
    target_vocab_size: int,
    *,
    expected_draft_vocab_size: int | None = None,
    t2d_mask: list[bool] | None = None,
    required_target_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Validate the offset format consumed by current vLLM EAGLE loaders."""
    errors: list[str] = []
    warnings: list[str] = []
    if expected_draft_vocab_size is not None and len(d2t_offsets) != expected_draft_vocab_size:
        errors.append(
            f"d2t length {len(d2t_offsets)} != draft vocab "
            f"{expected_draft_vocab_size}"
        )

    target_ids = [index + offset for index, offset in enumerate(d2t_offsets)]
    out_of_range = [
        target_id
        for target_id in target_ids
        if target_id < 0 or target_id >= target_vocab_size
    ]
    if out_of_range:
        errors.append(f"{len(out_of_range)} mapped target IDs are out of range")
    if len(set(target_ids)) != len(target_ids):
        errors.append("mapped target IDs are not unique")
    if target_ids != sorted(target_ids):
        warnings.append(
            "mapped target IDs are not sorted; this differs from AngelSlim's "
            "frequency-selection checkpoint format"
        )

    if t2d_mask is not None:
        if len(t2d_mask) != target_vocab_size:
            errors.append(
                f"t2d length {len(t2d_mask)} != target vocab {target_vocab_size}"
            )
        else:
            mask_ids = [index for index, included in enumerate(t2d_mask) if included]
            if mask_ids != target_ids:
                errors.append("t2d True positions do not exactly match d2t target IDs")

    required = sorted(set(required_target_ids or []))
    missing_required = [target_id for target_id in required if target_id not in target_ids]
    if missing_required:
        errors.append(f"required target IDs are absent: {missing_required}")

    return {
        "valid": not errors,
        "format": "offset (target_id = draft_id + d2t[draft_id])",
        "draft_vocab_size": len(d2t_offsets),
        "target_ids": target_ids,
        "min_target_id": min(target_ids, default=None),
        "max_target_id": max(target_ids, default=None),
        "required_target_ids": required,
        "missing_required_target_ids": missing_required,
        "errors": errors,
        "warnings": warnings,
    }


def profile_head_sizes(
    target_vocab_size: int,
    hidden_size: int,
    candidate_vocab_sizes: list[int],
    dtype_bytes: int,
) -> list[dict[str, Any]]:
    full_bytes = target_vocab_size * hidden_size * dtype_bytes
    profiles = []
    for vocab_size in candidate_vocab_sizes:
        if vocab_size <= 0 or vocab_size > target_vocab_size:
            raise ValueError(
                f"candidate vocab size {vocab_size} must be in [1, {target_vocab_size}]"
            )
        reduced_bytes = vocab_size * hidden_size * dtype_bytes
        profiles.append(
            {
                "vocab_size": vocab_size,
                "parameter_count": vocab_size * hidden_size,
                "head_bytes": reduced_bytes,
                "head_mib": round(reduced_bytes / 2**20, 3),
                "saved_mib": round((full_bytes - reduced_bytes) / 2**20, 3),
                "size_reduction_percent": round(
                    100.0 * (full_bytes - reduced_bytes) / full_bytes, 3
                ),
                "logit_columns_reduction_percent": round(
                    100.0 * (target_vocab_size - vocab_size) / target_vocab_size,
                    3,
                ),
            }
        )
    return profiles


def _literal_strings(path: Path, assignment_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        value = None
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == assignment_name
        ):
            value = node.value
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in node.targets
        ):
            value = node.value
        if value is not None:
            return [
                child.value
                for child in ast.walk(value)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            ]
    return []


def audit_vllm_source(vllm_source: Path, quantization: str | None) -> dict[str, Any]:
    spec_path = vllm_source / "vllm/config/speculative.py"
    gemma_path = vllm_source / "vllm/model_executor/models/gemma4_dflare.py"
    qwen_path = vllm_source / "vllm/model_executor/models/qwen3_dflash.py"
    quant_path = (
        vllm_source / "vllm/model_executor/layers/quantization/__init__.py"
    )
    missing = [
        str(path)
        for path in (spec_path, gemma_path, qwen_path, quant_path)
        if not path.is_file()
    ]
    if missing:
        return {
            "source_available": False,
            "safe_for_mapped_gemma4_dflare": False,
            "errors": [f"missing vLLM source file: {path}" for path in missing],
        }

    spec_source = spec_path.read_text(encoding="utf-8")
    gemma_source = gemma_path.read_text(encoding="utf-8")
    qwen_source = qwen_path.read_text(encoding="utf-8")
    quant_methods = _literal_strings(quant_path, "QuantizationMethods")

    hetero_draft_only = (
        "use_heterogeneous_vocab only works with method='draft_model'" in spec_source
    )
    gemma_mapping_supported = all(
        marker in gemma_source
        for marker in (
            "self.draft_id_to_target_id = nn.Parameter",
            'head_weights.append(("draft_id_to_target_id", weight))',
            "self.has_own_lm_head = True",
        )
    )
    gemma_mapping_disabled = not gemma_mapping_supported
    qwen_mapping_scatter = all(
        marker in qwen_source
        for marker in (
            'name = name.replace("d2t", "draft_id_to_target_id")',
            "targets = base + self.draft_id_to_target_id",
            "logits_new[:, targets] = logits",
        )
    )
    quantized_head_marker = (
        "ParallelLMHead(\n"
        "            vocab_size,\n"
        "            hidden_size,\n"
        "            quant_config="
    )
    gemma_head_receives_quant = quantized_head_marker in gemma_source

    quant_report = {
        "requested": quantization,
        "registered_methods": quant_methods,
        "recognized": quantization is None or quantization in quant_methods,
        "draft_layers_use_draft_quant_config": "get_draft_quant_config" in gemma_source,
        "gemma_dflare_lm_head_quantized": gemma_head_receives_quant,
        "scope": (
            "none requested"
            if quantization is None
            else "draft attention/MLP may use the draft quantization config; "
            "the Gemma DFlare lm_head remains unquantized"
        ),
    }

    blockers = []
    if gemma_mapping_disabled:
        blockers.append("Gemma DFlare does not allocate/load a d2t mapping")
    if not qwen_mapping_scatter:
        blockers.append("the expected EAGLE/DFlash d2t scatter reference is absent")
    if quantization is not None and quantization not in quant_methods:
        blockers.append(f"draft quantization method {quantization!r} is not registered")

    return {
        "source_available": True,
        "heterogeneous_vocab_is_draft_model_only": hetero_draft_only,
        "qwen_dflash_has_d2t_load_and_scatter": qwen_mapping_scatter,
        "gemma_dflare_mapping_supported": gemma_mapping_supported,
        "gemma_dflare_mapping_disabled": gemma_mapping_disabled,
        "safe_for_mapped_gemma4_dflare": not blockers,
        "blockers": blockers,
        "quantization": quant_report,
    }


def build_report(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    target = dict(config.get("target") or {})
    draft = dict(config.get("draft") or {})
    target_config_path = _resolve_path(target.get("config"), config_dir)
    draft_config_path = _resolve_path(draft.get("config"), config_dir)
    if target_config_path:
        loaded = _read_json(target_config_path)
        loaded.update({key: value for key, value in target.items() if key != "config"})
        target = loaded
    if draft_config_path:
        loaded = _read_json(draft_config_path)
        loaded.update({key: value for key, value in draft.items() if key != "config"})
        draft = loaded

    target_vocab_size = int(target["vocab_size"])
    hidden_size = int(target["hidden_size"])
    draft_input_vocab = int(draft.get("vocab_size", target_vocab_size))
    draft_output_vocab = int(draft.get("draft_vocab_size") or draft_input_vocab)
    candidates = [
        int(value)
        for value in config.get("candidate_vocab_sizes", [4096, 8192, 16384])
    ]
    dtype_bytes = int(config.get("dtype_bytes", 2))

    required_ids = []
    for source in (target, draft.get("dflare_config") or {}):
        for key in SPECIAL_ID_KEYS:
            value = source.get(key)
            if isinstance(value, int) and 0 <= value < target_vocab_size:
                required_ids.append(value)

    mapping_path = _resolve_path(config.get("mapping"), config_dir)
    mapping_report = {
        "provided": mapping_path is not None,
        "valid": False,
        "errors": ["no d2t mapping artifact was provided"],
    }
    if mapping_path is not None:
        d2t, t2d = _load_mapping(mapping_path)
        mapping_report = {
            "provided": True,
            "path": str(mapping_path),
            **validate_mapping(
                d2t,
                target_vocab_size,
                expected_draft_vocab_size=draft_output_vocab,
                t2d_mask=t2d,
                required_target_ids=required_ids,
            ),
        }
        # Keep reports compact; hashes/manifests should carry the full mapping.
        mapping_report.pop("target_ids", None)

    vllm_source = _resolve_path(config.get("vllm_source"), config_dir)
    quantization = config.get("draft_quantization")
    source_report = (
        audit_vllm_source(vllm_source, quantization)
        if vllm_source
        else {
            "source_available": False,
            "safe_for_mapped_gemma4_dflare": False,
            "errors": ["vllm_source was not configured"],
        }
    )

    findings = []
    if draft_input_vocab != target_vocab_size:
        findings.append(
            {
                "severity": "error",
                "code": "input-vocab-not-full",
                "message": (
                    f"draft input embedding vocab {draft_input_vocab} does not retain "
                    f"the full target vocab {target_vocab_size}"
                ),
            }
        )
    if draft_output_vocab < target_vocab_size and not mapping_report["valid"]:
        findings.append(
            {
                "severity": "error",
                "code": "reduced-head-mapping-invalid",
                "message": "a reduced output head requires a valid d2t artifact",
            }
        )
    if not source_report["safe_for_mapped_gemma4_dflare"]:
        findings.append(
            {
                "severity": "error",
                "code": "runtime-contract-missing",
                "message": "the checked-out Gemma DFlare runtime cannot consume mapped logits",
            }
        )
    if not target.get("tie_word_embeddings", False):
        findings.append(
            {
                "severity": "warning",
                "code": "target-head-not-declared-tied",
                "message": (
                    "target config does not declare tied embeddings; conversion would "
                    "need an explicit target lm_head artifact"
                ),
            }
        )

    safe = not any(item["severity"] == "error" for item in findings)
    return {
        "decision": "ready" if safe else "blocked",
        "implementation_branch": "A" if safe else "B",
        "summary": (
            "All mapped-head contracts validated."
            if safe
            else "Do not create a reduced Gemma-4 DFlare head with this runtime."
        ),
        "model": {
            "target_vocab_size": target_vocab_size,
            "draft_input_vocab_size": draft_input_vocab,
            "draft_output_vocab_size": draft_output_vocab,
            "hidden_size": hidden_size,
            "target_tied_embeddings": bool(target.get("tie_word_embeddings", False)),
            "required_target_ids": sorted(set(required_ids)),
        },
        "head_profiles": profile_head_sizes(
            target_vocab_size, hidden_size, candidates, dtype_bytes
        ),
        "mapping": mapping_report,
        "vllm": source_report,
        "findings": findings,
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"Decision: {report['decision'].upper()} (branch {report['implementation_branch']})")
    print(report["summary"])
    model = report["model"]
    print(
        "Vocab: target/input/output = "
        f"{model['target_vocab_size']}/{model['draft_input_vocab_size']}/"
        f"{model['draft_output_vocab_size']}"
    )
    for profile in report["head_profiles"]:
        print(
            f"  {profile['vocab_size']:>6} rows: {profile['head_mib']:>8.3f} MiB, "
            f"save {profile['saved_mib']:>8.3f} MiB "
            f"({profile['size_reduction_percent']:.3f}%)"
        )
    for finding in report["findings"]:
        print(f"{finding['severity'].upper()}: {finding['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--json", action="store_true", help="print JSON to stdout")
    parser.add_argument(
        "--require-safe",
        action="store_true",
        help="return status 2 unless the mapped runtime contract is complete",
    )
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    report = build_report(_read_json(config_path), config_path.parent)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    if args.json:
        print(rendered, end="")
    else:
        _print_human(report)
    return 2 if args.require_safe and report["decision"] != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
