#!/usr/bin/env python3
"""Plot epoch-level train and validation losses recorded by finetuning."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    log_file = args.log_file or ROOT / "phase3" / "checkpoints" / args.language / "training_log.jsonl"
    if not log_file.exists():
        raise FileNotFoundError(f"No finetuning log found at {log_file}. Run finetune_reasoning.py first.")
    rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    # If a run was resumed after an interrupted terminal session, retain the
    # best observed validation point per epoch rather than plotting duplicate
    # entries as separate experiments.
    best_by_epoch = {}
    for row in rows:
        if row["epoch"] not in best_by_epoch or row["val_loss"] < best_by_epoch[row["epoch"]]["val_loss"]:
            best_by_epoch[row["epoch"]] = row
    rows = [best_by_epoch[epoch] for epoch in sorted(best_by_epoch)]
    if not rows:
        raise ValueError("Finetuning log is empty.")
    output = args.output or ROOT / "report" / "images" / f"phase3_{args.language}_finetuning_loss.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in rows], marker="o", label="validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Answer-token cross-entropy")
    plt.title(f"Phase 3 {args.language.title()} reasoning finetuning")
    plt.xticks(epochs); plt.legend(); plt.tight_layout(); plt.savefig(output, dpi=180)
    print(f"Wrote {output}")

if __name__ == "__main__":
    main()
