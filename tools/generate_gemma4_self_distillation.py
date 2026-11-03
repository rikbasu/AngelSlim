#!/usr/bin/env python3
"""Generate Gemma target responses for a prompt-only conversation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--generation-batch-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    with args.prompts.open(encoding="utf-8") as prompt_file:
        records = [json.loads(line) for line in prompt_file if line.strip()]
    processor = AutoProcessor.from_pretrained(args.model)
    completed_ids = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as completed_file:
            completed_ids = {
                json.loads(line).get("id")
                for line in completed_file
                if line.strip()
            }
    kept_records = []
    prompt_token_ids = []
    max_prompt_tokens = args.max_model_len - args.max_new_tokens
    for record in records:
        if record.get("id") in completed_ids:
            continue
        messages = record["conversations"]
        token_ids = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if len(token_ids) <= max_prompt_tokens:
            kept_records.append(record)
            prompt_token_ids.append(token_ids)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
    )
    template_path = Path(args.model) / "chat_template.jinja"
    template_hash = (
        hashlib.sha256(template_path.read_bytes()).hexdigest()
        if template_path.exists()
        else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as output_file:
        for batch_start in range(
            0,
            len(prompt_token_ids),
            args.generation_batch_size,
        ):
            batch_records = kept_records[
                batch_start : batch_start + args.generation_batch_size
            ]
            batch_prompt_ids = prompt_token_ids[
                batch_start : batch_start + args.generation_batch_size
            ]
            outputs = llm.generate(
                [
                    TokensPrompt(prompt_token_ids=token_ids)
                    for token_ids in batch_prompt_ids
                ],
                SamplingParams(
                    temperature=args.temperature,
                    max_tokens=args.max_new_tokens,
                    seed=args.seed + batch_start,
                ),
            )
            for record, prompt_ids, generated in zip(
                batch_records,
                batch_prompt_ids,
                outputs,
            ):
                response = generated.outputs[0]
                result = {
                    **record,
                    "conversations": [
                        *record["conversations"],
                        {"role": "assistant", "content": response.text},
                    ],
                    "target_model": args.model,
                    "generation_dtype": "bfloat16",
                    "temperature": args.temperature,
                    "seed": args.seed + batch_start,
                    "chat_template_sha256": template_hash,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": list(response.token_ids),
                    "eos_terminated": response.finish_reason == "stop",
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                generated_count += 1
            output_file.flush()

    print(
        json.dumps(
            {
                "input_prompts": len(records),
                "already_completed": len(completed_ids),
                "generated_prompts": generated_count,
                "filtered_for_length": (
                    len(records) - len(completed_ids) - len(kept_records)
                ),
                "output": str(args.output),
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
