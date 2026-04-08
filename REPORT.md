# Final report: biased addition corpus and naive LM

## 1. Goal

Train the same small causal Transformer on **intentionally skewed** addition data (pairs **(a, b)** with **a** and **b** in **[1, 100]**), then measure validation and test metrics. The point is to show how **sampling bias** in the corpus shapes what the model sees and how well it fits under teacher forcing and greedy completion.

## 2. Dataset bias (what we changed)

We replaced uniform sampling over unique pairs with **`small_operands` bias**:

| Parameter | Value | Meaning |
|-----------|--------|---------|
| `bias_mode` | `small_operands` | With probability `bias_mix`, draw both **a** and **b** uniformly from **[1, bias_region_max]**; otherwise draw each operand uniformly from **[1, 100]**. |
| `bias_mix` | `0.88` | 88% of *proposed* draws use the small region. |
| `bias_region_max` | `35` | “Small” operands are capped at 35. |

**Formalization:** Each line is still `a+b=c` with no spaces. Splits remain **70% / 15% / 15%** (train / val / test). We still require **unique (a, b)** pairs until we reach `n_examples = 5000`.

**Empirical statistics** on the final 5000 unique pairs (see `data/manifest.json`):

| Statistic | Value | Comment |
|-----------|--------|---------|
| Fraction with both **a ≤ 35** and **b ≤ 35** | **0.245** | Lower than 88% because (i) many biased draws collide when deduplicating, and (ii) the sampler must fill the set with extra full-range pairs. |
| Fraction with **a ≤ 35** | **0.438** | Asymmetric marginal: first operand more often small. |
| Mean of **a + b** | **92.47** | Still high overall because large pairs remain in the set. |

So the bias is **real but softened** by the uniqueness constraint—a realistic lesson: *even with biased sampling, deduplication and coverage requirements change the final distribution.*

## 3. Model and training

- **Architecture:** `TinyCausalTransformer` (2 layers, hidden size **64**, 4 heads, FFN 128, max length 32), next-token cross-entropy with padding ignored.
- **Hardware:** CPU (`python train.py ... --cpu`).
- **Run:** `train.py --epochs 35 --batch-size 128 --runs-dir runs_biased`.

Artifacts:

- Weights: `runs_biased/best.pt` (best validation loss).
- Curves: `runs_biased/history.json`, plot `runs_biased/learning_curves.png`.
- Table: `runs_biased/metrics_table.csv`.

## 4. Results (biased run)

Numbers from `runs_biased/test_metrics.json` after re-evaluation.

### 4.1 Teacher-forced metrics (standard LM evaluation)

| Metric | Validation | Test |
|--------|------------|------|
| Cross-entropy loss (lower is better) | 1.560 | 1.550 |
| Token accuracy | 0.450 | 0.456 |

### 4.2 Greedy completion (prefix `a+b=`)

| Metric | Validation | Test |
|--------|------------|------|
| Exact line match (full string) | 0.27% (2/750) | 0.67% (5/750) |
| Numeric correctness (parsed sum) | 0.27% | 0.67% |
| Parseable greedy completions | 100% | 100% |

**Interpretation:** Teacher-forced accuracy is **much higher** than exact greedy completion because the latter requires **multi-step** correctness without teacher tokens. The biased corpus slightly improved loss vs. earlier informal uniform runs on smaller data; exact-match rates remain small but **non-zero** on this run.

### 4.3 Bias and fairness angle

- Training and test both come from the **same biased generative process**, so metrics are **in-distribution**. They do **not** measure fairness on a uniform reference grid where both operands range from 1 to 100 independently.
- To study **mismatch**, you would train on the biased file and evaluate on a **uniform** `test.txt` (or stratify by operand size). That would typically show weaker performance on **large a** and **large b** cases if the model overfits the small-operand regime.

## 5. How to reproduce

```bash
cd naive_lm
source .venv/bin/activate   # or your venv

python build_dataset.py --out-dir data --n-examples 5000 --seed 42 \
  --bias-mode small_operands --bias-mix 0.88 --bias-region-max 35

python train.py --epochs 35 --batch-size 128 --cpu --runs-dir runs_biased

python evaluate.py --checkpoint runs_biased/best.pt --data-dir data \
  --out runs_biased/test_metrics.json --cpu

python plot_metrics.py --history runs_biased/history.json \
  --out-png runs_biased/learning_curves.png --out-csv runs_biased/metrics_table.csv
```

Uniform baseline for comparison: add `--bias-mode uniform` to `build_dataset.py` and use a separate `--runs-dir` (e.g. `runs_uniform`).

## 6. Short conclusion

We introduced **deliberate sampling bias** toward smaller operands (`small_operands`, mix 0.88, cap 35). After deduplication, about **24.5%** of pairs had both operands **≤ 35**, still skewing the corpus compared to uniform sampling. The retrained model achieved **~0.45** token accuracy (teacher-forced) and **non-zero** greedy exact matches on the biased test set, while greedy decoding remains hard for this tiny LM. The pipeline demonstrates how **data construction** (including bias and uniqueness) directly affects what metrics mean and how they should be reported.
