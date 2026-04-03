"""PyTorch Dataset for line-delimited equations."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import EOS_ID, PAD_ID, encode


class EquationDataset(Dataset):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        raw = self.path.read_text(encoding="utf-8").strip().splitlines()
        self.lines = [ln.strip() for ln in raw if ln.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = encode(self.lines[idx], append_eos=True)
        return torch.tensor(ids, dtype=torch.long)


def collate_batch(pad_id: int = PAD_ID):
    def _collate(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_t = max(x.numel() for x in batch)
        out = []
        for x in batch:
            if x.numel() < max_t:
                pad = torch.full((max_t - x.numel(),), pad_id, dtype=torch.long)
                x = torch.cat([x, pad], dim=0)
            out.append(x)
        seq = torch.stack(out, dim=0)
        # Next-token targets: predict token at t from prefix ending at t-1
        inp = seq[:, :-1].contiguous()
        tgt = seq[:, 1:].contiguous()
        return inp, tgt

    return _collate
