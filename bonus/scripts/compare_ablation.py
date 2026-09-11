#!/usr/bin/env python3
"""Side-by-Side Comparison Script: Standard Model H vs. Ablated Model H (No Positional Embeddings).

Generates:
1. Comparative summary table (PPL, BPB, BLEU-4, chrF, ROUGE-L, Diversity, Attention)
2. Comparison charts saved in bonus/images/ and report/images/bonus/
3. Machine-readable JSON summary in bonus/results/metrics_comparison.json
"""
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Standard Phase 2 Model H baseline metrics
STANDARD_MODEL_H = {
    "model_name": "Standard Model H (RoPE)",
    "position_encoding": "Rotary Position Embeddings (RoPE)",
    "parameters": 36837888,
    "cross_entropy_loss": 4.0162,
    "perplexity": 55.4911,
    "bits_per_byte": 5.7942,
    "bleu_4": 3.51,
    "chrf": 9.19,
    "rouge_l": 0.0612,
    "distinct_1": 0.1321,
    "distinct_2": 0.2671,
    "repetition_rate": 0.6738,
    "layer_0_entropy": 1.2540,
    "layer_0_distance": 3.1240,
    "layer_5_entropy": 1.1890,
    "layer_5_distance": 7.4210
}

def generate_comparison(ablation_json_path, output_json_path, output_fig_paths):
    if not os.path.exists(ablation_json_path):
        print(f"Warning: Ablation metrics file {ablation_json_path} not found.")
        return

    with open(ablation_json_path, "r", encoding="utf-8") as f:
        ablation_data = json.load(f)

    intrin = ablation_data.get("intrinsic_metrics", {})
    gen = ablation_data.get("generation_metrics", {})
    attn = ablation_data.get("attention_metrics", {})

    ablated_summary = {
        "model_name": "Ablated Model H (No Pos)",
        "position_encoding": "None (Ablated)",
        "parameters": 36837888,
        "cross_entropy_loss": intrin.get("cross_entropy_loss", 0.0),
        "perplexity": intrin.get("perplexity", 0.0),
        "bits_per_byte": intrin.get("bits_per_byte", 0.0),
        "bleu_4": gen.get("bleu_4", 0.0),
        "chrf": gen.get("chrf", 0.0),
        "rouge_l": gen.get("rouge_l", 0.0),
        "distinct_1": gen.get("distinct_1", 0.0),
        "distinct_2": gen.get("distinct_2", 0.0),
        "repetition_rate": gen.get("repetition_rate", 0.0),
        "layer_0_entropy": attn.get("layer_0", {}).get("mean_entropy", 0.0),
        "layer_0_distance": attn.get("layer_0", {}).get("mean_distance", 0.0),
        "layer_5_entropy": attn.get("layer_5", {}).get("mean_entropy", 0.0),
        "layer_5_distance": attn.get("layer_5", {}).get("mean_distance", 0.0)
    }

    comparison_report = {
        "standard_model": STANDARD_MODEL_H,
        "ablated_model": ablated_summary,
        "deltas": {
            "perplexity_delta": round(ablated_summary["perplexity"] - STANDARD_MODEL_H["perplexity"], 4),
            "bits_per_byte_delta": round(ablated_summary["bits_per_byte"] - STANDARD_MODEL_H["bits_per_byte"], 4),
            "bleu_delta": round(ablated_summary["bleu_4"] - STANDARD_MODEL_H["bleu_4"], 2),
            "repetition_rate_delta": round(ablated_summary["repetition_rate"] - STANDARD_MODEL_H["repetition_rate"], 4),
            "layer_0_distance_delta": round(ablated_summary["layer_0_distance"] - STANDARD_MODEL_H["layer_0_distance"], 4),
            "layer_5_distance_delta": round(ablated_summary["layer_5_distance"] - STANDARD_MODEL_H["layer_5_distance"], 4)
        }
    }

    # Print clean markdown table
    print("\n" + "=" * 80)
    print("PHASE 2 STANDARD MODEL vs. NO-POSITION ABLATED MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Standard (RoPE)':<20} | {'Ablated (No Pos)':<20}")
    print("-" * 80)
    print(f"{'Parameters':<30} | {STANDARD_MODEL_H['parameters']:<20,} | {ablated_summary['parameters']:<20,}")
    print(f"{'Cross-Entropy Loss':<30} | {STANDARD_MODEL_H['cross_entropy_loss']:<20.4f} | {ablated_summary['cross_entropy_loss']:<20.4f}")
    print(f"{'Perplexity (PPL)':<30} | {STANDARD_MODEL_H['perplexity']:<20.2f} | {ablated_summary['perplexity']:<20.2f}")
    print(f"{'Bits-Per-Byte (BPB)':<30} | {STANDARD_MODEL_H['bits_per_byte']:<20.4f} | {ablated_summary['bits_per_byte']:<20.4f}")
    print(f"{'BLEU-4':<30} | {STANDARD_MODEL_H['bleu_4']:<20.2f} | {ablated_summary['bleu_4']:<20.2f}")
    print(f"{'chrF':<30} | {STANDARD_MODEL_H['chrf']:<20.2f} | {ablated_summary['chrf']:<20.2f}")
    print(f"{'ROUGE-L':<30} | {STANDARD_MODEL_H['rouge_l']:<20.4f} | {ablated_summary['rouge_l']:<20.4f}")
    print(f"{'Distinct-1':<30} | {STANDARD_MODEL_H['distinct_1']:<20.4f} | {ablated_summary['distinct_1']:<20.4f}")
    print(f"{'Distinct-2':<30} | {STANDARD_MODEL_H['distinct_2']:<20.4f} | {ablated_summary['distinct_2']:<20.4f}")
    print(f"{'Repetition Rate':<30} | {STANDARD_MODEL_H['repetition_rate']:<20.4f} | {ablated_summary['repetition_rate']:<20.4f}")
    print(f"{'Layer 0 Attention Distance':<30} | {STANDARD_MODEL_H['layer_0_distance']:<20.4f} | {ablated_summary['layer_0_distance']:<20.4f}")
    print(f"{'Layer 5 Attention Distance':<30} | {STANDARD_MODEL_H['layer_5_distance']:<20.4f} | {ablated_summary['layer_5_distance']:<20.4f}")
    print("=" * 80)

    # Save JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, ensure_ascii=False, indent=2)
    print(f"Saved comparison JSON to {output_json_path}")

    # Plot comparison bar chart
    metrics_to_plot = ["BLEU-4", "chrF", "Distinct-1 (%)", "Distinct-2 (%)", "Repetition (%)"]
    std_vals = [
        STANDARD_MODEL_H["bleu_4"],
        STANDARD_MODEL_H["chrf"],
        STANDARD_MODEL_H["distinct_1"] * 100,
        STANDARD_MODEL_H["distinct_2"] * 100,
        STANDARD_MODEL_H["repetition_rate"] * 100
    ]
    abl_vals = [
        ablated_summary["bleu_4"],
        ablated_summary["chrf"],
        ablated_summary["distinct_1"] * 100,
        ablated_summary["distinct_2"] * 100,
        ablated_summary["repetition_rate"] * 100
    ]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, std_vals, width, label="Standard Model H (RoPE)", color="#2ca02c")
    plt.bar(x + width/2, abl_vals, width, label="Ablated Model H (No Pos)", color="#d62728")
    plt.title("Generation & Fluency Metrics: Standard vs. No-Position Ablation")
    plt.ylabel("Score / Percentage")
    plt.xticks(x, metrics_to_plot)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    for p in output_fig_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=180)
        print(f"Saved comparison figure to {p}")
    plt.close()

if __name__ == "__main__":
    ablation_json = ROOT / "bonus" / "results" / "evaluation_metrics.json"
    output_json = ROOT / "bonus" / "results" / "metrics_comparison.json"
    fig_paths = [
        ROOT / "bonus" / "images" / "standard_vs_nopos_comparison.png",
        ROOT / "report" / "images" / "bonus" / "standard_vs_nopos_comparison.png"
    ]
    generate_comparison(str(ablation_json), str(output_json), [str(p) for p in fig_paths])
