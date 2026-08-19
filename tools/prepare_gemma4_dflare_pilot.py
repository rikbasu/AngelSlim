#!/usr/bin/env python3
"""Convert multimodal planner messages into text-only DFlare pilot data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def convert_record(record: dict, record_id: int) -> dict | None:
    source_messages = record.get("messages") or record.get("conversations")
    if not isinstance(source_messages, list):
        return None
    conversations = []
    for message in source_messages:
        role = message.get("role")
        content = text_content(message.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            conversations.append({"role": role, "content": content})
    if not any(item["role"] == "assistant" for item in conversations):
        return None
    return {"id": f"planner-{record_id:06d}", "conversations": conversations}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--eval-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    for index, line in enumerate(
        args.input.read_text(encoding="utf-8").splitlines()
    ):
        if line.strip():
            converted = convert_record(json.loads(line), index)
            if converted is not None:
                records.append(converted)
    random.Random(args.seed).shuffle(records)
    eval_size = min(args.eval_size, max(1, len(records) // 10))
    eval_records = records[:eval_size]
    train_records = records[eval_size:]
    write_jsonl(args.train_output, train_records)
    write_jsonl(args.eval_output, eval_records)
    print(
        json.dumps(
            {
                "input_records": len(records),
                "train_records": len(train_records),
                "eval_records": len(eval_records),
                "images_removed": True,
                "purpose": "pipeline integration only",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
