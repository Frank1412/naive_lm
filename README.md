# Naive LM: integer addition (1–100)

Small **decoder-style causal Transformer** trained on synthetic lines `a+b=c` using only digits, `+`, and `=`. Designed for **CPU** training and inference.

## Formalization

| Item | Definition |
|------|------------|
| Task | Predict the full string `a+b=c` left-to-right (next-token LM). |
| Operands | Integers \(a, b \in [1, 100]\). |
| Vocabulary | 14 tokens: `0`–`9`, `+`, `=`, `<PAD>`, `<EOS>`. |
| Line format | One equation per line, no spaces (see `data/manifest.json`). |
| Splits | Train / validation / test (default 70% / 15% / 15%). |

**Bias note:** Pairs \((a,b)\) are sampled uniformly at random (with unique pairs). Real usage or human-written math can follow other distributions; a model trained only on this corpus may not reflect that and can amplify skew if you later mix data sources.

## Environment

```bash
cd /Users/shouzhifang/Documents/Syracuse/735/naive_lm
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1) Build the dataset

```bash
python3 build_dataset.py --out-dir data --n-examples 5000 --seed 42
```

This writes `data/train.txt`, `data/val.txt`, `data/test.txt`, `data/manifest.json`, and `data/tokenizer_note.txt`.

## 2) Train (CPU)

```bash
python3 train.py --epochs 40 --batch-size 128 --cpu --runs-dir runs
```

Checkpoints: `runs/best.pt` (lowest validation loss). Training curves: `runs/history.json`.

## 3) Learning curves and table

```bash
python3 plot_metrics.py --history runs/history.json --out-png runs/learning_curves.png --out-csv runs/metrics_table.csv
```

Open `runs/learning_curves.png` (loss vs epoch; validation token accuracy on the right axis). `runs/metrics_table.csv` is a flat table for reports.

## 4) Evaluate

```bash
python3 evaluate.py --checkpoint runs/best.pt --data-dir data --out runs/test_metrics.json --cpu
```

Metrics include teacher-forced **token accuracy** and **loss**, plus **greedy completion** after the prefix `a+b=` (exact line match and numeric correctness).

## Model size (default)

Roughly **2 layers**, **d_model=64**, **4 heads**, **FFN=128**, **max sequence length 32** — small enough for laptops without a GPU.

## Files

| File | Role |
|------|------|
| `tokenizer.py` | Character-level encoding and equation formatting |
| `model.py` | `TinyCausalTransformer` (TransformerEncoder + causal attention) |
| `data_module.py` | PyTorch `Dataset` and batch collation |
| `build_dataset.py` | Generate splits |
| `train.py` | Training loop |
| `evaluate.py` | Metrics and JSON report |
| `plot_metrics.py` | PNG curve + CSV table |
# naive_lm
