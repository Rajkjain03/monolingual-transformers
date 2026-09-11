#!/usr/bin/env python3
"""Pretraining script for the Ablated Decoder-Only Transformer (No Positional Embeddings).

Supports:
- Memory-mapped streaming dataset loading
- Automatic Mixed Precision (AMP) with torch.cuda.amp
- Gradient accumulation for simulated large batch sizes
- Full resume-capable checkpoint saving (weights, optimizer, scheduler, step, config)
- Regular validation logging and loss curve plotting
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bonus" / "scripts"))
sys.path.insert(0, str(ROOT / "hindi" / "scripts"))

from dataloader import get_dataloader
from model_no_pos import LanguageModelNoPos

def save_checkpoint(model, optimizer, scheduler, step, loss, config, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else {},
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else {},
        'step': step,
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"[Checkpointer] Saved resumable checkpoint to {path}")

def evaluate_val_loss(model, val_dl, device, max_steps=50):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
                _, loss = model(x, targets=y)
            total_loss += loss.item()
            steps += 1
            if steps >= max_steps:
                break
    model.train()
    return total_loss / max(1, steps)

def plot_loss_curves(log_file, output_image):
    if not os.path.exists(log_file):
        return
    steps, train_losses, val_steps, val_losses = [], [], [], []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "train_loss" in data:
                steps.append(data["step"])
                train_losses.append(data["train_loss"])
            if "val_loss" in data:
                val_steps.append(data["step"])
                val_losses.append(data["val_loss"])
    
    if not steps:
        return
        
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="Train Loss (Cross-Entropy)", color="#1f77b4", alpha=0.8)
    if val_losses:
        plt.plot(val_steps, val_losses, label="Validation Loss", color="#ff7f0e", marker="o", linewidth=2)
    plt.title("Ablation: Pretraining Loss Curve (No Positional Embeddings)")
    plt.xlabel("Optimizer Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, dpi=180)
    plt.close()
    print(f"[Plotter] Saved loss curve to {output_image}")

def main():
    parser = argparse.ArgumentParser(description="Train Transformer with No Positional Embeddings")
    parser.add_argument("--config", type=str, default=str(ROOT / "bonus" / "configs" / "config_no_pos.json"))
    parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--val-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"============================================================")
    print(f"Bonus Ablation: Pretraining LM without Positional Embeddings")
    print(f"Device: {device} | Max Steps: {args.max_steps} | LR: {args.lr}")
    print(f"Effective Batch Size: {args.batch_size * args.grad_accum_steps}")
    print(f"============================================================")

    model = LanguageModelNoPos(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        hidden_dim=config["hidden_dim"],
        max_seq_len=config["max_seq_len"]
    ).to(device)

    print(f"Model Parameters: {model.get_num_params():,} (Same parameter count as standard Model H)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps), eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    start_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    train_bin = str(ROOT / config.get("train_data", "hindi/data/hindi_train.bin"))
    val_bin = str(ROOT / config.get("val_data", "hindi/data/hindi_val.bin"))

    train_dl = get_dataloader(train_bin, batch_size=args.batch_size, seq_len=config["max_seq_len"], shuffle=True)
    val_dl = get_dataloader(val_bin, batch_size=args.batch_size, seq_len=config["max_seq_len"], shuffle=False)

    ckpt_dir = ROOT / "bonus" / "checkpoints"
    results_dir = ROOT / "bonus" / "results"
    images_dir = ROOT / "bonus" / "images"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "training_log.jsonl"
    loss_plot_path = images_dir / "loss_curve_no_pos.png"

    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    accum_count = 0
    step = start_step

    train_iter = iter(train_dl)
    start_time = time.time()

    while step < args.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss = model(x, targets=y)
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
        else:
            logits, loss = model(x, targets=y)
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()

        running_loss += loss.item()
        accum_count += 1

        if accum_count == args.grad_accum_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            step += 1
            accum_count = 0
            avg_step_loss = running_loss / args.grad_accum_steps
            running_loss = 0.0

            # Log progress
            if step % 20 == 0 or step == args.max_steps:
                elapsed = time.time() - start_time
                print(f"[Step {step:4d}/{args.max_steps}] Loss: {avg_step_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Elapsed: {elapsed:.1f}s")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "train_loss": avg_step_loss, "lr": scheduler.get_last_lr()[0]}) + "\n")

            # Periodic validation
            if step % args.val_interval == 0 or step == args.max_steps:
                val_loss = evaluate_val_loss(model, val_dl, device)
                val_ppl = math.exp(val_loss)
                print(f"--- [Validation @ Step {step}] Loss: {val_loss:.4f} | PPL: {val_ppl:.2f} ---")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "val_loss": val_loss, "val_ppl": val_ppl}) + "\n")
                plot_loss_curves(log_path, loss_plot_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, step, val_loss, config, ckpt_dir / "best.pt")

            # Periodic saving of last.pt
            if step % args.save_interval == 0 or step == args.max_steps:
                save_checkpoint(model, optimizer, scheduler, step, avg_step_loss, config, ckpt_dir / "last.pt")

    print(f"Pretraining complete. Best checkpoint saved at {ckpt_dir / 'best.pt'}")
    plot_loss_curves(log_path, loss_plot_path)

if __name__ == "__main__":
    main()
