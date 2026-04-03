#!/usr/bin/env python3
"""Evaluate exact-match completion, token accuracy (teacher-forced), and numeric correctness."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_module import EquationDataset, collate_batch
from model import TinyCausalTransformer
from tokenizer import EOS_ID, PAD_ID, VOCAB_SIZE, decode, encode

_EQ = re.compile(r"^(\d+)\+(\d+)=(\d+)$")


def parse_equation(line: str) -> tuple[int, int, int] | None:
    m = _EQ.match(line.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def teacher_forced_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = 0
    with torch.no_grad():
        for inp, tgt in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            logits = model(inp)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            total_loss += loss.item()
            n_batches += 1
            pred = logits.argmax(dim=-1)
            mask = tgt != PAD_ID
            total_correct += (pred[mask] == tgt[mask]).sum().item()
            total_count += mask.sum().item()
    avg_loss = total_loss / max(n_batches, 1)
    tok_acc = total_correct / max(total_count, 1)
    return avg_loss, tok_acc


def greedy_completion_exact(
    model: TinyCausalTransformer,
    lines: list[str],
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (exact_line_match_rate, numeric_match_rate, parseable_rate)."""
    model.eval()
    exact = 0
    numeric_ok = 0
    parsed = 0
    for line in lines:
        g = parse_equation(line)
        if g is None:
            continue
        a, b, gold_sum = g
        prefix = f"{a}+{b}="
        prefix_ids = encode(prefix, append_eos=False)
        out_ids = model.generate_greedy(prefix_ids, max_new_tokens=16, eos_id=EOS_ID)
        pred_line = decode(out_ids, strip_special=True)
        if pred_line == line.strip():
            exact += 1
        pg = parse_equation(pred_line)
        if pg is not None:
            parsed += 1
            pa, pb, ps = pg
            if pa == a and pb == b and ps == a + b:
                numeric_ok += 1
    n = len(lines)
    return exact / n, numeric_ok / n, parsed / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/best.pt"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("runs/test_metrics.json"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = TinyCausalTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg.get("dropout", 0.1),
        max_len=cfg["max_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    collate = collate_batch(PAD_ID)
    val_ds = EquationDataset(args.data_dir / "val.txt")
    test_ds = EquationDataset(args.data_dir / "test.txt")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    val_loss, val_tok = teacher_forced_metrics(model, val_loader, device)
    test_loss, test_tok = teacher_forced_metrics(model, test_loader, device)

    val_lines = val_ds.lines
    test_lines = test_ds.lines
    vex, vnum, vpar = greedy_completion_exact(model, val_lines, device)
    tex, tnum, tpar = greedy_completion_exact(model, test_lines, device)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "metrics": {
            "val_loss_teacher": val_loss,
            "test_loss_teacher": test_loss,
            "val_token_accuracy": val_tok,
            "test_token_accuracy": test_tok,
            "val_exact_line_match": vex,
            "test_exact_line_match": tex,
            "val_numeric_correct_given_prefix": vnum,
            "test_numeric_correct_given_prefix": tnum,
            "val_parseable_completion_rate": vpar,
            "test_parseable_completion_rate": tpar,
        },
        "notes": {
            "exact_line_match": "Greedy decode after prefix 'a+b='; full string must match.",
            "numeric_correct": "Parsed a,b,sum from greedy completion; a,b fixed from prefix.",
            "teacher_metrics": "Teacher-forced next-token loss/accuracy on padded batches.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
