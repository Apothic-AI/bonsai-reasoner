#!/usr/bin/env python3
"""Normalize reasoning datasets into chat JSONL for Bonsai LoRA SFT."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset


SOURCE_CONFIGS: dict[str, dict[str, str | None]] = {
    "bespoke-stratos": {
        "dataset": "bespokelabs/Bespoke-Stratos-17k",
        "config": None,
        "split": "train",
        "kind": "bespoke_conversations",
    },
    "gpt54-xhigh-2000": {
        "dataset": "vanty120/Gpt-5.4-Xhigh-Reasoning-2000x",
        "config": None,
        "split": "train",
        "kind": "thinking_response",
    },
    "gpt54-xhigh-750": {
        "dataset": "vanty120/Gpt-5.4-Xhigh-Reasoning-750x",
        "config": None,
        "split": "train",
        "kind": "thinking_response",
    },
    "gpt54-step-by-step": {
        "dataset": "Roman1111111/gpt-5.4-step-by-step-reasoning",
        "config": None,
        "split": "train",
        "kind": "question_answer",
    },
    "gpt54-xhigh-dalisoft": {
        "dataset": "dalisoft/gpt-5.4-xhigh-reasoning-550x",
        "config": None,
        "split": "train",
        "kind": "messages",
    },
    "gpt54-bird-sql": {
        "dataset": "wenyupapa/BIRD-Verified-CoT-2462-GPT5.4",
        "config": None,
        "split": "train",
        "kind": "bird_sql",
    },
    "mot-math": {
        "dataset": "open-r1/Mixture-of-Thoughts",
        "config": "math",
        "split": "train",
        "kind": "messages",
    },
    "mot-code": {
        "dataset": "open-r1/Mixture-of-Thoughts",
        "config": "code",
        "split": "train",
        "kind": "messages",
    },
    "mot-science": {
        "dataset": "open-r1/Mixture-of-Thoughts",
        "config": "science",
        "split": "train",
        "kind": "messages",
    },
    "withinus-gpt55-recursive": {
        "dataset": "WithinUsAI/GPT5.5_thinking_max_distill_god_seed_25K",
        "config": None,
        "split": "train",
        "kind": "instruction_input_output",
    },
    "dolly-gpt55": {
        "dataset": "Bialy17/databricks-dolly-1k-GPT5.5",
        "config": None,
        "split": "train",
        "kind": "instruction_output",
    },
    "versper-orpo-chosen": {
        "dataset": "versperai/Versper-V1-Evo-ORPO-GPT5.5-Think-Recursive-25k",
        "config": None,
        "split": "train",
        "kind": "preference_chosen",
    },
}

DEFAULT_REJECT_PATTERN = re.compile(
    r"GPT-5\.5 Thinking Max Distill|I am GPT-5\.5|Versper-V1-Evo|"
    r"Recursive Seed AI|god-level|intelligence explosion|self-evolving",
    re.IGNORECASE,
)


def normalize_role(role: str) -> str:
    role = role.lower().strip()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    raise ValueError(f"unsupported role: {role!r}")


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        role = normalize_role(str(message.get("role", message.get("from", ""))))
        content = str(message.get("content", message.get("value", ""))).strip()
        if content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def normalize_row(row: dict[str, Any], kind: str) -> list[dict[str, str]]:
    if kind == "bespoke_conversations":
        messages = []
        system = str(row.get("system") or "").strip()
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(clean_messages(list(row["conversations"])))
        return messages

    if kind == "messages":
        return clean_messages(list(row["messages"]))

    if kind == "thinking_response":
        instruction = str(row.get("instruction") or "").strip()
        thinking = str(row.get("thinking") or "").strip()
        response = str(row.get("response") or "").strip()
        assistant = response
        if thinking:
            assistant = f"<think>\n{thinking}\n</think>\n\n{response}"
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": assistant}]

    if kind == "question_answer":
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    if kind == "bird_sql":
        question = str(row.get("question") or "").strip()
        evidence = str(row.get("evidence") or "").strip()
        schema = row.get("schema") or {}
        ddl = str(schema.get("ddl") if isinstance(schema, dict) else "").strip()
        reasoning = str(row.get("reasoning") or "").strip()
        sql = str(row.get("SQL") or "").strip()
        user_parts = ["Write the SQL query for this database question.", f"Question:\n{question}"]
        if evidence:
            user_parts.append(f"Evidence:\n{evidence}")
        if ddl:
            user_parts.append(f"Schema:\n{ddl}")
        assistant = reasoning if not sql else f"{reasoning}\n\nSQL:\n{sql}"
        return [{"role": "user", "content": "\n\n".join(user_parts)}, {"role": "assistant", "content": assistant}]

    if kind == "instruction_input_output":
        instruction = str(row.get("instruction") or "").strip()
        input_text = str(row.get("input") or "").strip()
        output = str(row.get("output") or "").strip()
        user_text = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
        return [{"role": "user", "content": user_text}, {"role": "assistant", "content": output}]

    if kind == "instruction_output":
        instruction = str(row.get("instruction") or "").strip()
        output = str(row.get("output") or "").strip()
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]

    if kind == "preference_chosen":
        prompt = str(row.get("prompt") or "").strip()
        chosen = str(row.get("chosen") or "").strip()
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]

    raise ValueError(f"unsupported dataset kind: {kind}")


def has_user_and_assistant(messages: list[dict[str, str]]) -> bool:
    roles = {message["role"] for message in messages}
    return "user" in roles and "assistant" in roles


def rejected(messages: list[dict[str, str]], reject_pattern: re.Pattern[str] | None, max_chars: int) -> bool:
    if not has_user_and_assistant(messages):
        return True
    joined = "\n".join(message["content"] for message in messages)
    if len(joined) > max_chars:
        return True
    return bool(reject_pattern and reject_pattern.search(joined))


def iter_source(source: str, streaming: bool) -> Iterable[dict[str, Any]]:
    config = SOURCE_CONFIGS[source]
    dataset_name = str(config["dataset"])
    dataset_config = config["config"]
    split = str(config["split"])
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    return load_dataset(dataset_name, split=split, streaming=streaming)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", choices=sorted(SOURCE_CONFIGS), default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--allow-identity-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    sources = args.source or ["bespoke-stratos"]
    reject_pattern = None if args.allow_identity_data else DEFAULT_REJECT_PATTERN
    target_total = args.limit + args.eval_size

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        kind = str(SOURCE_CONFIGS[source]["kind"])
        for row in iter_source(source, streaming=not args.no_streaming):
            try:
                messages = normalize_row(dict(row), kind)
            except Exception:
                continue
            if rejected(messages, reject_pattern, args.max_chars):
                continue
            key = "\n".join(message["content"] for message in messages[-2:])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"source": source, "messages": messages})
            if len(rows) >= target_total:
                break
        if len(rows) >= target_total:
            break

    random.shuffle(rows)
    eval_rows = rows[: args.eval_size]
    train_rows = rows[args.eval_size : args.eval_size + args.limit]
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    print(
        json.dumps(
            {
                "sources": sources,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_output": str(args.train_output),
                "eval_output": str(args.eval_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
