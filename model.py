"""Tiny causal Transformer (decoder-style LM) for short arithmetic strings."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tokenizer import PAD_ID, VOCAB_SIZE


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.size(1)
        x = x + self.pe[:, :t]
        return self.dropout(x)


class TinyCausalTransformer(nn.Module):
    """Stacked self-attention encoder with causal mask (GPT-style next-token LM)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        t = x.size(1)
        if t > self.max_len:
            raise ValueError(f"sequence length {t} > max_len {self.max_len}")
        h = self.embed(x) * math.sqrt(self.d_model)
        h = self.pos(h)
        pad = x == PAD_ID
        # Causal: True = position cannot attend (upper triangle, excluding diagonal).
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad)
        h = self.ln_f(h)
        return self.lm_head(h)

    @torch.no_grad()
    def generate_greedy(
        self,
        prefix_ids: list[int],
        max_new_tokens: int = 16,
        eos_id: int | None = None,
    ) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        out = list(prefix_ids)
        for _ in range(max_new_tokens):
            if ids.size(1) > self.max_len:
                break
            logits = self.forward(ids)
            next_id = int(logits[0, -1].argmax().item())
            out.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        return out
