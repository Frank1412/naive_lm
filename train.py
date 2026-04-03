#!/usr/bin/env python3
"""Train tiny causal LM on CPU (or CUDA if available)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_module import EquationDataset, collate_batch
from model import TinyCausalTransformer
from tokenizer import EOS_ID, PAD_ID, VOCAB_SIZE


def token_accuracy(logits: torch.Tensor, tgt: torch.Tensor) -> float:
    """Argmax next-token accuracy on non-padding targets."""
    pred = logits.argmax(dim=-1)
    mask = tgt != PAD_ID
    if mask.sum().item() == 0:
        return 0.0
    return (pred[mask] == tgt[mask]).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for inp, tgt in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(inp)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        total_acc += token_accuracy(logits, tgt)
        n_batches += 1
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    train_ds = EquationDataset(args.data_dir / "train.txt")
    val_ds = EquationDataset(args.data_dir / "val.txt")
    collate = collate_batch(PAD_ID)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    model = TinyCausalTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_token_acc": [],
    }
    best_val = float("inf")
    best_path = args.runs_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt, criterion)
        va_loss, va_acc = run_epoch(model, val_loader, device, None, criterion)
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_token_acc"].append(va_acc)
        print(
            f"epoch {epoch:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
            f"val_tok_acc={va_acc:.4f}"
        )
        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_layers": args.layers,
                        "dim_feedforward": args.ff,
                        "dropout": args.dropout,
                        "max_len": args.max_len,
                        "vocab_size": VOCAB_SIZE,
                    },
                },
                best_path,
            )

    with open(args.runs_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history -> {args.runs_dir / 'history.json'}")
    print(f"Best checkpoint -> {best_path} (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
