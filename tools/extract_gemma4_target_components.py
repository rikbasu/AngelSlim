#!/usr/bin/env python3
"""Extract Gemma-4's tied embedding/lm-head for offline DFlare training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


EMBED_KEY = "model.language_model.embed_tokens.weight"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source_weights = args.model / "model.safetensors"
    if not source_weights.exists():
        raise FileNotFoundError(source_weights)
    args.output.mkdir(parents=True, exist_ok=True)

    with safe_open(source_weights, framework="pt", device="cpu") as source:
        if EMBED_KEY not in source.keys():
            raise KeyError(f"{EMBED_KEY} not found in {source_weights}")
        embedding = source.get_tensor(EMBED_KEY)
    save_file({EMBED_KEY: embedding}, args.output / "model.safetensors")
    shutil.copy2(args.model / "config.json", args.output / "config.json")

    result = {
        "source": str(args.model),
        "output": str(args.output),
        "weight_key": EMBED_KEY,
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "tied_lm_head": True,
    }
    (args.output / "extraction.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
