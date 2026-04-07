#!/usr/bin/env python3
"""Build train / val / test text files for integer addition a+b=c (operands in [1, 100])."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tokenizer import equation_line


def sample_pair(
    rng: random.Random,
    low: int,
    high: int,
    bias_mode: str,
    bias_mix: float,
    bias_region_max: int,
) -> tuple[int, int]:
    """Draw one (a, b). For biased modes, see manifest description."""
    if bias_mode == "uniform":
        return rng.randint(low, high), rng.randint(low, high)
    if bias_mode == "small_operands":
        # Oversample problems where both operands lie in a smaller sub-range (e.g. "easy" drills).
        if rng.random() < bias_mix:
            cap = max(low, min(bias_region_max, high))
            return rng.randint(low, cap), rng.randint(low, cap)
        return rng.randint(low, high), rng.randint(low, high)
    if bias_mode == "first_operand_small":
        # Asymmetric bias: first addend usually small, second uniform (simulates "n + something" worksheets).
        if rng.random() < bias_mix:
            cap = max(low, min(bias_region_max, high))
            a = rng.randint(low, cap)
            b = rng.randint(low, high)
            return a, b
        return rng.randint(low, high), rng.randint(low, high)
    raise ValueError(f"Unknown bias_mode: {bias_mode}")


def pair_stats(pairs: set[tuple[int, int]], bias_region_max: int) -> dict:
    both_small = sum(1 for a, b in pairs if a <= bias_region_max and b <= bias_region_max)
    first_small = sum(1 for a, b in pairs if a <= bias_region_max)
    return {
        "n_unique_pairs": len(pairs),
        "fraction_both_operands_le_bias_cap": both_small / max(len(pairs), 1),
        "fraction_first_operand_le_bias_cap": first_small / max(len(pairs), 1),
        "mean_sum": sum(a + b for a, b in pairs) / max(len(pairs), 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate addition dataset splits.")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--n-examples", type=int, default=5000, help="Total unique equations (before split).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument(
        "--bias-mode",
        choices=("uniform", "small_operands", "first_operand_small"),
        default="small_operands",
        help="uniform: i.i.d. uniform pairs; small_operands: mix toward both a,b in [1,bias_region_max]; "
        "first_operand_small: mix toward small a, b uniform.",
    )
    p.add_argument(
        "--bias-mix",
        type=float,
        default=0.88,
        help="For biased modes: probability each sample uses the biased draw (else full-range uniform).",
    )
    p.add_argument(
        "--bias-region-max",
        type=int,
        default=35,
        help="Upper bound for the 'small' operand region (inclusive, clipped to [1,100]).",
    )
    args = p.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < -1e-9:
        raise SystemExit("train_ratio + val_ratio must be <= 1")

    rng = random.Random(args.seed)
    low, high = 1, 100
    pairs: set[tuple[int, int]] = set()
    max_attempts = args.n_examples * 80
    attempts = 0
    while len(pairs) < args.n_examples and attempts < max_attempts:
        a, b = sample_pair(rng, low, high, args.bias_mode, args.bias_mix, args.bias_region_max)
        pairs.add((a, b))
        attempts += 1
    if len(pairs) < args.n_examples:
        raise SystemExit(
            f"Could only collect {len(pairs)} unique pairs; try lowering n_examples or bias_mix."
        )

    lines = [equation_line(a, b) for (a, b) in pairs]
    rng.shuffle(lines)

    n = len(lines)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train_lines = lines[:n_train]
    val_lines = lines[n_train : n_train + n_val]
    test_lines = lines[n_train + n_val :]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    (out / "test.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")

    stats = pair_stats(pairs, args.bias_region_max)
    manifest = {
        "name": "integer_addition_1_100",
        "n_examples": n,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(test_ratio, 6),
        "operand_range": [low, high],
        "format_description": 'Each line: "a+b=c" with integers, no spaces.',
        "splits": {
            "train": len(train_lines),
            "val": len(val_lines),
            "test": len(test_lines),
        },
        "bias": {
            "mode": args.bias_mode,
            "bias_mix": args.bias_mix,
            "bias_region_max": args.bias_region_max,
            "description": {
                "uniform": "Each (a,b) drawn uniformly from [1,100]^2 until n unique pairs.",
                "small_operands": "With probability bias_mix, both a and b drawn uniformly from [1,bias_region_max]; "
                "otherwise uniform on full [1,100]^2. Creates oversampling of smaller operands.",
                "first_operand_small": "With probability bias_mix, a from [1,bias_region_max] and b uniform [1,100]; "
                "else full uniform. Emphasizes small leading addends.",
            }[args.bias_mode],
        },
        "empirical_pair_statistics": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    note = (
        f"Vocab size: 14\n"
        f"Symbols: ['0'..'9', '+', '=', '<PAD>', '<EOS>']\n"
        f"Example: '3+4=7' -> token ids via tokenizer.encode()\n"
        f"Bias mode: {args.bias_mode} (see manifest.json)\n"
    )
    (out / "tokenizer_note.txt").write_text(note, encoding="utf-8")
    print(f"Wrote {out}/train.txt ({len(train_lines)}), val.txt ({len(val_lines)}), test.txt ({len(test_lines)})")
    print(f"Manifest: {out / 'manifest.json'}")
    print(f"Pair stats: {stats}")


if __name__ == "__main__":
    main()
