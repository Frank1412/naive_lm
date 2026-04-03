"""Character-level tokenizer: digits, +, =, plus PAD and EOS (minimal vocab)."""

from __future__ import annotations

# Vocab order fixed for reproducibility (14 tokens).
_DIGITS_AND_OPS = "0123456789+="  # indices 0..11
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SPECIAL = [PAD_TOKEN, EOS_TOKEN]

# id -> char/special name for debugging
ID_TO_SYM: dict[int, str] = {i: c for i, c in enumerate(_DIGITS_AND_OPS)}
ID_TO_SYM[12] = PAD_TOKEN
ID_TO_SYM[13] = EOS_TOKEN

SYM_TO_ID: dict[str, int] = {v: k for k, v in ID_TO_SYM.items()}

PAD_ID = SYM_TO_ID[PAD_TOKEN]
EOS_ID = SYM_TO_ID[EOS_TOKEN]
VOCAB_SIZE = len(ID_TO_SYM)


def encode(text: str, append_eos: bool = True) -> list[int]:
    out: list[int] = []
    for ch in text:
        if ch not in SYM_TO_ID:
            raise ValueError(f"Invalid character {ch!r} in {text!r}")
        out.append(SYM_TO_ID[ch])
    if append_eos:
        out.append(EOS_ID)
    return out


def decode(ids: list[int] | list, strip_special: bool = True) -> str:
    parts: list[str] = []
    for i in ids:
        if i == PAD_ID and strip_special:
            continue
        if i == EOS_ID:
            break
        parts.append(ID_TO_SYM[int(i)])
    return "".join(parts)


def equation_line(a: int, b: int) -> str:
    """Format a single supervised line: a+b=c (no spaces)."""
    return f"{a}+{b}={a + b}"
