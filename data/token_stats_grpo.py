#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from collections import OrderedDict

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-1.7B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute token stats for GRPO-style jsonl (question + answer1~5 + solution)."
    )
    parser.add_argument("--data-path", required=True, type=str)
    return parser.parse_args()


def pick_answer(record):
    """Return the text for the correct answer using 'solution'."""
    solution = record.get("solution")
    if solution is None:
        return None
    solution = str(solution).strip()

    # Accept numeric labels (1-5), zero-based, or letter labels.
    idx = None
    if solution.isdigit():
        idx = int(solution)
    else:
        letters = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        idx = letters.get(solution.lower())

    if idx is None:
        return None

    # Support zero-based (0-4) or one-based (1-5) indexes.
    key = f"answer{idx}" if idx in range(1, 10) else None
    if key and key in record:
        return record[key]

    # if idx looked 0-based try idx+1
    if idx in range(0, 9):
        alt_key = f"answer{idx + 1}"
        if alt_key in record:
            return record[alt_key]
    return None


def load_messages(path):
    dataset = []
    with open(path, "r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("question")
            answer = pick_answer(record)
            if not question or not answer:
                continue
            dataset.append(
                {
                    "messages": [
                        {"role": "user", "content": question.strip()},
                        {"role": "assistant", "content": answer.strip()},
                    ]
                }
            )
    return dataset


def percentile(data, q):
    if not data:
        return 0
    k = (len(data) - 1) * (q / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "<|im_end|>" in tokenizer.get_vocab():
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|im_end|>"
    tokenizer.padding_side = "right"
    return tokenizer


def describe(lengths):
    lengths_sorted = sorted(lengths)
    total_tokens = sum(lengths_sorted)
    summary = {
        "count": len(lengths_sorted),
        "sum": total_tokens,
        "min": lengths_sorted[0] if lengths_sorted else 0,
        "max": lengths_sorted[-1] if lengths_sorted else 0,
        "mean": round(statistics.fmean(lengths_sorted), 2) if lengths_sorted else 0,
        "median": percentile(lengths_sorted, 50),
        "p75": percentile(lengths_sorted, 75),
        "p90": percentile(lengths_sorted, 90),
        "p95": percentile(lengths_sorted, 95),
        "p99": percentile(lengths_sorted, 99),
    }
    return summary


def build_bins():
    edges = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    bins = OrderedDict()
    prev = 0
    for edge in edges:
        bins[(prev, edge)] = f"{prev + 1}-{edge}"
        prev = edge
    bins[(edges[-1], math.inf)] = f">{edges[-1]}"
    return bins


def bin_counts(lengths):
    bins = build_bins()
    ordered_counts = OrderedDict((label, 0) for label in bins.values())
    for value in lengths:
        for (low, high), label in bins.items():
            if low < value <= high:
                ordered_counts[label] += 1
                break
    total = sum(ordered_counts.values())
    return ordered_counts, total


def main():
    args = parse_args()
    tokenizer = prepare_tokenizer()
    dataset = load_messages(args.data_path)

    lengths = []
    for sample in dataset:
        token_ids = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        lengths.append(len(token_ids))

    stats = describe(lengths)
    bins, total = bin_counts(lengths)

    print("=== Token Statistics (GRPO) ===")
    for key, value in stats.items():
        print(f"{key:>7}: {value}")

    print("\n=== Distribution (count / percentage) ===")
    for label, count in bins.items():
        pct = (count / total) * 100 if total else 0
        print(f"{label:>10}: {count:>6} ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
