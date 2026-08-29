import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import os
import json
import time
from dataloader import get_dataloader
from model import LanguageModel

def save_checkpoint(model, optimizer, scheduler, step, loss, config, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

def train():
    config = {
        "vocab_size": 16000,
        "dim": 512,
        "n_layers": 6,
        "n_heads": 8,
        "hidden_dim": 2048,
        "max_seq_len": 512,
        "batch_size": 4, # Reduced for 4GB VRAM
        "lr": 5e-4,
        "epochs": 1,
        "grad_accum_steps": 16, # Effective batch size 64
        "val_interval": 1000,
        "save_interval": 5000,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"Training on device: {config['device']}")
    
    model = LanguageModel(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        hidden_dim=config["hidden_dim"],
        max_seq_len=config["max_seq_len"]
    ).to(config["device"])
    
    print(f"Model Parameters: {model.get_num_params():,}")
    
    train_dl = get_dataloader(
        "nepali/data/nepali_train.bin", 
        batch_size=config["batch_size"], 
        seq_len=config["max_seq_len"],
        shuffle=True
    )
    
    val_dl = get_dataloader(
        "nepali/data/nepali_val.bin", 
        batch_size=config["batch_size"], 
        seq_len=config["max_seq_len"],
        shuffle=False
    )
    
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    
    # Approx total steps
    total_steps = len(train_dl) // config["grad_accum_steps"] * config["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    scaler = torch.amp.GradScaler('cuda') if config["device"] == "cuda" else None
    
    os.makedirs("nepali/checkpoints", exist_ok=True)
    
    model.train()
    step = 0
    optimizer.zero_grad()
    
    start_t = time.time()
    for epoch in range(config["epochs"]):
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(config["device"]), y.to(config["device"])
            
            if scaler is not None:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits, loss = model(x, targets=y)
                    loss = loss / config["grad_accum_steps"]
                scaler.scale(loss).backward()
            else:
                logits, loss = model(x, targets=y)
                loss = loss / config["grad_accum_steps"]
                loss.backward()
            
            if (i + 1) % config["grad_accum_steps"] == 0:
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
                
                if step % 10 == 0:
                    dt = time.time() - start_t
                    print(f"Step {step} | Loss: {loss.item() * config['grad_accum_steps']:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {dt:.2f}s")
                    start_t = time.time()
                
                if step % config["val_interval"] == 0:
                    model.eval()
                    val_loss = 0
                    val_steps = 0
                    with torch.no_grad():
                        for vx, vy in val_dl:
                            vx, vy = vx.to(config["device"]), vy.to(config["device"])
                            _, vloss = model(vx, targets=vy)
                            val_loss += vloss.item()
                            val_steps += 1
                            if val_steps >= 50: # subset eval
                                break
                    print(f"--- Validation Loss: {val_loss/val_steps:.4f} ---")
                    model.train()
                    
                if step % config["save_interval"] == 0:
                    save_checkpoint(model, optimizer, scheduler, step, loss.item() * config["grad_accum_steps"], config, f"nepali/checkpoints/ckpt_{step}.pt")
                    
    save_checkpoint(model, optimizer, scheduler, step, loss.item() * config["grad_accum_steps"], config, f"nepali/checkpoints/final.pt")

if __name__ == "__main__":
    train()
