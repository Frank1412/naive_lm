#!/usr/bin/env python3
"""Build train / val / test text files for integer addition a+b=c (operands in [1, 100])."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tokenizer import equation_line


def main() -> None:
    p = argparse.ArgumentParser(description="Generate addition dataset splits.")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--n-examples", type=int, default=5000, help="Total unique equations (before split).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    args = p.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < -1e-9:
        raise SystemExit("train_ratio + val_ratio must be <= 1")

    rng = random.Random(args.seed)
    low, high = 1, 100
    pairs: set[tuple[int, int]] = set()
    # Sample until we have enough unique (a,b) pairs
    max_attempts = args.n_examples * 50
    attempts = 0
    while len(pairs) < args.n_examples and attempts < max_attempts:
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        pairs.add((a, b))
        attempts += 1
    if len(pairs) < args.n_examples:
        raise SystemExit(f"Could only collect {len(pairs)} unique pairs; increase range or lower n_examples")

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
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    note = (
        f"Vocab size: 14\n"
        f"Symbols: ['0'..'9', '+', '=', '<PAD>', '<EOS>']\n"
        f"Example: '3+4=7' -> token ids via tokenizer.encode()\n"
    )
    (out / "tokenizer_note.txt").write_text(note, encoding="utf-8")
    print(f"Wrote {out}/train.txt ({len(train_lines)}), val.txt ({len(val_lines)}), test.txt ({len(test_lines)})")
    print(f"Manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
