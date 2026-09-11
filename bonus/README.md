# Bonus Part: Transformer Ablation Study (No Positional Embeddings)

This directory contains the complete implementation, training pipeline, full Phase 2 evaluation suite, empirical verification scripts, and scientific comparison report for the **No Positional Embeddings Ablation Study** as described in the project specification (`LMA_Individual_Project_v1.pdf`, Section "Bonus (optional)").

---

## Directory Structure

```
bonus/
├── configs/
│   └── config_no_pos.json               # Hyperparameters for the ablated model
├── scripts/
│   ├── model_no_pos.py                  # Decoder-only Transformer with RoPE removed
│   ├── train_no_pos.py                  # Resumable pretraining script (AMP + Accumulation)
│   ├── evaluate_no_pos.py               # Full Phase 2 evaluation suite (PPL, BPB, BLEU, chrF, ROUGE-L, Diversity, Attention)
│   ├── verify_order_invariance.py       # Empirical test of distance & word-order invariance
│   └── compare_ablation.py              # Side-by-side metrics table & visual comparison generator
├── results/
│   ├── training_log.jsonl               # Step-by-step loss and learning rate logs
│   ├── evaluation_metrics.json          # Complete JSON output of Phase 2 evaluation suite
│   └── metrics_comparison.json          # Comparative metrics (Standard vs. Ablated)
├── images/
│   ├── attn_L0_H0_no_pos.png            # Attention heatmaps (Layer 0, Head 0)
│   ├── attn_L0_H2_no_pos.png            # Attention heatmaps (Layer 0, Head 2)
│   ├── attn_L5_H0_no_pos.png            # Attention heatmaps (Layer 5, Head 0)
│   ├── attn_L5_H2_no_pos.png            # Attention heatmaps (Layer 5, Head 2)
│   ├── loss_curve_no_pos.png            # Pretraining loss curve (Train vs. Val)
│   └── standard_vs_nopos_comparison.png # Comparison bar chart
├── checkpoints/
│   ├── best.pt                          # Best validation checkpoint (resumable)
│   └── last.pt                          # Latest checkpoint (resumable)
├── report.md                            # Complete scientific Bonus Report
└── README.md                            # This guide
```

*(Note: The root directory also provides an alias `bonous/` redirecting to this directory.)*

---

## Quickstart & Reproduction Steps

All scripts run out-of-the-box using the project virtual environment (`./lma_env/bin/python`):

### 1. Empirically Verify Order & Distance Invariance
Demonstrates mathematically and empirically that self-attention dot products without positional embeddings have zero distance-dependent attenuation:
```bash
./lma_env/bin/python bonus/scripts/verify_order_invariance.py
```

### 2. Pretrain the Ablated Model
Train the 36.8M parameter ablated model from scratch on the Hindi monolingual corpus with Automatic Mixed Precision (AMP) and Gradient Accumulation:
```bash
./lma_env/bin/python bonus/scripts/train_no_pos.py --max-steps 100 --val-interval 50 --save-interval 100
```
*Checkpoints are saved to `bonus/checkpoints/best.pt` and `bonus/checkpoints/last.pt`. To resume an interrupted training run:*
```bash
./lma_env/bin/python bonus/scripts/train_no_pos.py --resume bonus/checkpoints/last.pt
```

### 3. Run the Full Phase 2 Evaluation Suite
Evaluates intrinsic metrics (Cross-Entropy, PPL, BPB), generation quality (BLEU-4, chrF, ROUGE-L), diversity diagnostics (Distinct-1, Distinct-2, Repetition Rate across temperatures 0.0, 0.5, 1.0, 1.5), and outputs attention heatmaps:
```bash
./lma_env/bin/python bonus/scripts/evaluate_no_pos.py
```

### 4. Generate Side-by-Side Model Comparison & Plots
Generates the comparison tables and bar charts comparing the Standard Model H with the Ablated Model H:
```bash
./lma_env/bin/python bonus/scripts/compare_ablation.py
```

---

## Summary of Results

| Metric | Standard Model H (RoPE) | Ablated Model H (No Pos) | Consequence of Removing Position |
|---|:---:|:---:|---|
| **Parameters** | 36,837,888 | 36,837,888 | Identical capacity & parameter count |
| **Cross-Entropy Loss** | **4.0162** | 7.8385 | Loss floor severely elevated |
| **Perplexity (PPL)** | **55.49** | 2,536.42 | $\sim 45.7\times$ higher perplexity |
| **Bits-Per-Byte (BPB)** | **5.7942** | 11.3086 | $+5.5144$ bits/byte |
| **BLEU-4** | **3.51** | 1.09 | Significant drop in n-gram precision |
| **chrF** | **9.19** | 0.88 | Drop in subword/character matching |
| **ROUGE-L** | **0.0612** | 0.0224 | Collapse in longest common subsequence |
| **Distinct-1** | **0.1321** | 0.0160 | Extreme vocabulary repetition |
| **Distinct-2** | **0.2671** | 0.0221 | Collapse of bigram diversity |
| **Repetition Rate** | **0.6738** | **0.8853** | Surges to 88.5% (catastrophic attractor loops) |
| **Late Attention Dist (L5)** | **7.4210** | 2.2653 | Inability to form long-range syntactic paths |

---

## Detailed Report

For the full theoretical breakdown, analysis of syntactic role inversion, and discussion of why greedy generation degenerates into periodic attractor loops, please read [`bonus/report.md`](report.md) (also mirrored at [`report/bonus_report.md`](../report/bonus_report.md)).
