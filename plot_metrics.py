#!/usr/bin/env python3
"""Plot learning curves from runs/history.json and optionally print a summary table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, default=Path("runs/history.json"))
    ap.add_argument("--out-png", type=Path, default=Path("runs/learning_curves.png"))
    ap.add_argument("--out-csv", type=Path, default=Path("runs/metrics_table.csv"))
    args = ap.parse_args()

    data = json.loads(args.history.read_text(encoding="utf-8"))
    epochs = data["epoch"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    val_acc = data.get("val_token_acc", [])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, label="train loss", color="tab:blue")
    ax1.plot(epochs, val_loss, label="val loss", color="tab:orange")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.grid(True, alpha=0.3)

    if val_acc:
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_acc, label="val token acc", color="tab:green", linestyle="--")
        ax2.set_ylabel("val token accuracy")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved plot -> {args.out_png}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_token_acc"])
        for i, e in enumerate(epochs):
            row = [e, train_loss[i], val_loss[i]]
            row.append(val_acc[i] if i < len(val_acc) else "")
            w.writerow(row)
    print(f"Saved table -> {args.out_csv}")


if __name__ == "__main__":
    main()
