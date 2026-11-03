#!/usr/bin/env python3
"""Distributed Hugging Face self-distillation for Gemma prompt corpora."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, AutoTokenizer


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-file", type=Path)
    parser.add_argument("--max-model-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records-per-rank", type=int)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    records = load_jsonl(args.prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part_path = args.output.with_suffix(f".rank{rank:03d}.jsonl")
    existing_records = []
    if args.resume_file:
        existing_records.extend(load_jsonl(args.resume_file))
    for part in sorted(args.output.parent.glob(f"{args.output.stem}.rank*.jsonl")):
        existing_records.extend(load_jsonl(part))
    completed_ids = {record.get("id") for record in existing_records}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="eager",
    )
    model.eval()

    candidates = []
    max_prompt_tokens = args.max_model_length - args.max_new_tokens
    for index, record in enumerate(records):
        if record.get("id") in completed_ids or index % world_size != rank:
            continue
        token_ids = tokenizer.apply_chat_template(
            record["conversations"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if len(token_ids) <= max_prompt_tokens:
            candidates.append((record, token_ids))
            if (
                args.max_records_per_rank is not None
                and len(candidates) >= args.max_records_per_rank
            ):
                break

    generated_count = 0
    with part_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(candidates), args.batch_size):
            batch = candidates[start : start + args.batch_size]
            encoded = tokenizer.pad(
                {"input_ids": [token_ids for _, token_ids in batch]},
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(local_rank) for key, value in encoded.items()}
            batch_seed = args.seed + rank * 1_000_000 + start
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed(batch_seed)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
            prompt_width = encoded["input_ids"].shape[1]
            for (record, prompt_ids), sequence in zip(batch, generated):
                response_ids = sequence[prompt_width:].tolist()
                result = {
                    **record,
                    "conversations": [
                        *record["conversations"],
                        {
                            "role": "assistant",
                            "content": tokenizer.decode(
                                response_ids,
                                skip_special_tokens=True,
                            ),
                        },
                    ],
                    "target_model": args.model,
                    "generation_dtype": "bfloat16",
                    "temperature": args.temperature,
                    "seed": batch_seed,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": response_ids,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                generated_count += 1
            output.flush()
            print(
                f"rank={rank} generated={generated_count}/{len(candidates)}",
                flush=True,
            )

    dist.barrier()
    if rank == 0:
        merged = {}
        if args.resume_file:
            for record in load_jsonl(args.resume_file):
                merged[record["id"]] = record
        for part in sorted(args.output.parent.glob(f"{args.output.stem}.rank*.jsonl")):
            for record in load_jsonl(part):
                merged[record["id"]] = record
        order = {record["id"]: index for index, record in enumerate(records)}
        temporary = args.output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in sorted(
                merged.values(),
                key=lambda item: order.get(item["id"], len(order)),
            ):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "input_prompts": len(records),
                    "completed": len(merged),
                },
                indent=2,
            )
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
