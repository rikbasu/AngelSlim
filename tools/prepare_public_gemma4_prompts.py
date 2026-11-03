#!/usr/bin/env python3
"""Build a deterministic mixed public-prompt corpus for Gemma self-distillation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def take_shuffled(name, config, split, count, seed):
    dataset = load_dataset(name, config, split=split, streaming=True)
    return list(dataset.shuffle(seed=seed, buffer_size=10_000).take(count))


def write_record(output, record_id, source, prompt):
    output.write(
        json.dumps(
            {
                "id": record_id,
                "source": source,
                "conversations": [{"role": "user", "content": prompt.strip()}],
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def extract_user_prompt(messages):
    for message in messages or []:
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
    return ""


def word_ngrams(text, size):
    """N-grams of `text`, falling back to the whole prompt when it is shorter.

    Without the fallback a prompt with fewer than `size` words yields no n-grams
    at all and silently bypasses overlap filtering. That is not hypothetical:
    the 60-prompt eval set contains 10 prompts under 32 words, and all 10 of
    them leaked into a training corpus built with --overlap-ngram-size 32 while
    the filter reported zero overlaps.
    """
    words = re.findall(r"\w+", text.lower())
    if not words:
        return set()
    if len(words) < size:
        return {tuple(words)}
    return {
        tuple(words[index : index + size])
        for index in range(len(words) - size + 1)
    }


def load_excluded_ngrams(path, size):
    if path is None:
        return set()
    excluded = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("conversations") or record.get("messages") or []
            prompt = extract_user_prompt(messages)
            if not prompt and record.get("turns"):
                prompt = str(record["turns"][0])
            excluded.update(word_ngrams(prompt, size))
    return excluded


def prepare_dflare_mix(count, seed):
    code_alpaca_count = min(20_000, count // 5)
    nemotron_count = count - code_alpaca_count
    math_count = nemotron_count // 3
    code_count = nemotron_count // 3
    chat_count = nemotron_count - math_count - code_count
    records = []
    for split, split_count, split_seed in (
        ("math", math_count, seed),
        ("code", code_count, seed + 1),
        ("chat", chat_count, seed + 2),
    ):
        rows = take_shuffled(
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            "SFT",
            split,
            split_count,
            split_seed,
        )
        for index, row in enumerate(rows):
            prompt = extract_user_prompt(row.get("messages"))
            if prompt:
                records.append(
                    (
                        f"nemotron-{split}-{index:06d}",
                        f"nvidia/Nemotron-Post-Training-Dataset-v2:SFT/{split}",
                        prompt,
                    )
                )
    rows = take_shuffled(
        "sahil2801/CodeAlpaca-20k",
        None,
        "train",
        code_alpaca_count,
        seed + 3,
    )
    for index, row in enumerate(rows):
        prompt = row.get("prompt") or row.get("instruction", "")
        if row.get("input"):
            prompt += "\n\n" + row["input"]
        if prompt:
            records.append(
                (
                    f"codealpaca-{index:06d}",
                    "sahil2801/CodeAlpaca-20k",
                    prompt,
                )
            )
    return records


def prepare_dflare_diverse_mix(count, seed):
    """Balanced math / code / chat mix.

    The existing recipes draw only from CodeAlpaca and UltraChat, and six
    training arms established that adding more of that corpus cannot raise
    acceptance at any learning rate. This mix swaps in GSM8K (no mathematical
    content exists in the current corpus at all) and a different code
    instruction set, keeping UltraChat for conversation, so the prompt
    distribution is the variable under test rather than its volume.
    """
    math_count = count // 3 + (1 if count % 3 else 0)
    code_count = count // 3 + (1 if count % 3 == 2 else 0)
    chat_count = count - math_count - code_count
    records = []

    for index, row in enumerate(
        take_shuffled("openai/gsm8k", "main", "train", math_count, seed)
    ):
        prompt = row.get("question")
        if prompt:
            records.append((f"math-{index:06d}", "openai/gsm8k", prompt))

    for index, row in enumerate(
        take_shuffled(
            "iamtarun/python_code_instructions_18k_alpaca",
            None,
            "train",
            code_count,
            seed + 1,
        )
    ):
        prompt = row.get("instruction", "")
        if row.get("input"):
            prompt += "\n\n" + row["input"]
        if prompt:
            records.append(
                (
                    f"pycode-{index:06d}",
                    "iamtarun/python_code_instructions_18k_alpaca",
                    prompt,
                )
            )

    for index, row in enumerate(
        take_shuffled(
            "HuggingFaceH4/ultrachat_200k", None, "train_sft", chat_count, seed + 2
        )
    ):
        prompt = row.get("prompt") or extract_user_prompt(row.get("messages"))
        if prompt:
            records.append(
                (f"chat-{index:06d}", "HuggingFaceH4/ultrachat_200k", prompt)
            )

    return records


def prepare_dflare_open_mix(count, seed):
    code_alpaca_count = min(20_000, max(1, int(count * 0.4)))
    ultrachat_count = count - code_alpaca_count
    records = []
    rows = take_shuffled(
        "sahil2801/CodeAlpaca-20k",
        None,
        "train",
        code_alpaca_count,
        seed,
    )
    for index, row in enumerate(rows):
        prompt = row.get("prompt") or row.get("instruction", "")
        if row.get("input"):
            prompt += "\n\n" + row["input"]
        if prompt:
            records.append(
                (
                    f"codealpaca-{index:06d}",
                    "sahil2801/CodeAlpaca-20k",
                    prompt,
                )
            )
    rows = take_shuffled(
        "HuggingFaceH4/ultrachat_200k",
        None,
        "train_sft",
        ultrachat_count,
        seed + 1,
    )
    for index, row in enumerate(rows):
        prompt = row.get("prompt") or extract_user_prompt(row.get("messages"))
        if prompt:
            records.append(
                (
                    f"ultrachat-{index:06d}",
                    "HuggingFaceH4/ultrachat_200k",
                    prompt,
                )
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recipe",
        choices=["legacy", "dflare", "dflare_open", "dflare_diverse"],
        default="legacy",
    )
    parser.add_argument("--exclude-overlap-file", type=Path)
    parser.add_argument("--overlap-ngram-size", type=int, default=32)
    args = parser.parse_args()

    if args.recipe in {"dflare", "dflare_open", "dflare_diverse"}:
        excluded = load_excluded_ngrams(
            args.exclude_overlap_file,
            args.overlap_ngram_size,
        )
        builders = {
            "dflare": prepare_dflare_mix,
            "dflare_open": prepare_dflare_open_mix,
            "dflare_diverse": prepare_dflare_diverse_mix,
        }
        records = builders[args.recipe](args.count, args.seed)
        kept = []
        filtered = 0
        seen_prompts = set()
        for record_id, source, prompt in records:
            normalized = " ".join(prompt.split())
            if not normalized or normalized in seen_prompts:
                filtered += 1
                continue
            if excluded and word_ngrams(prompt, args.overlap_ngram_size) & excluded:
                filtered += 1
                continue
            seen_prompts.add(normalized)
            kept.append((record_id, source, prompt))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            for record in kept:
                write_record(output, *record)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "requested": args.count,
                    "written": len(kept),
                    "filtered_overlap_or_duplicate": filtered,
                    "recipe": args.recipe,
                    "seed": args.seed,
                },
                indent=2,
            )
        )
        return 0

    math_count = args.count // 3 + (1 if args.count % 3 else 0)
    code_count = args.count // 3 + (1 if args.count % 3 == 2 else 0)
    chat_count = args.count - math_count - code_count

    math_rows = take_shuffled(
        "openai/gsm8k", "main", "train", math_count, args.seed
    )
    code_rows = take_shuffled(
        "iamtarun/python_code_instructions_18k_alpaca",
        None,
        "train",
        code_count,
        args.seed + 1,
    )
    chat_rows = take_shuffled(
        "HuggingFaceH4/ultrachat_200k",
        None,
        "train_sft",
        chat_count,
        args.seed + 2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for index, row in enumerate(math_rows):
            write_record(output, f"math-{index:04d}", "openai/gsm8k", row["question"])
        for index, row in enumerate(code_rows):
            prompt = row.get("prompt") or (
                row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
            )
            write_record(
                output,
                f"code-{index:04d}",
                "iamtarun/python_code_instructions_18k_alpaca",
                prompt,
            )
        for index, row in enumerate(chat_rows):
            write_record(
                output,
                f"chat-{index:04d}",
                "HuggingFaceH4/ultrachat_200k",
                row["prompt"],
            )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "count": args.count,
                "math": math_count,
                "code": code_count,
                "chat": chat_count,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
