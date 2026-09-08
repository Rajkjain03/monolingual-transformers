#!/usr/bin/env python3
"""Full supervised reasoning finetuning from a Phase 2 checkpoint.

Only answer tokens contribute to loss.  Checkpoints retain model, optimiser,
scheduler, epoch and step state, so an interrupted run can be resumed exactly.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]

class ReasoningDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        self.sp, self.max_seq_len = tokenizer, max_seq_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prefix = row["prompt"] + "\n" + row["answer_prefix"]
        prefix_ids = self.sp.encode_as_ids(prefix)
        answer_ids = self.sp.encode_as_ids(row["answer"]) + [self.sp.eos_id()]
        ids = (prefix_ids + answer_ids)[:self.max_seq_len + 1]
        # Predict ids[t+1]; mask prompt continuation, retaining answer + EOS.
        x = ids[:-1]
        y = ids[1:]
        answer_start = max(0, len(prefix_ids) - 1)
        labels = [-100] * min(answer_start, len(y)) + y[answer_start:]
        return torch.tensor(x), torch.tensor(labels)

def collate(batch, pad_id):
    max_len = max(len(x) for x, _ in batch)
    xs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    ys = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, :len(x)], ys[i, :len(y)] = x, y
    return xs, ys

def build_model(language, device):
    language_dir = ROOT / language
    sys.path.insert(0, str(language_dir / "scripts"))
    from model import LanguageModel, precompute_freqs_cis
    config = json.loads((language_dir / "configs" / "config.json").read_text())
    # Avoid initializing random 30M-parameter weights immediately before
    # replacing them from the checkpoint.  This matters on CPU/RAM-limited
    # machines: normal construction temporarily holds two full models.
    with torch.device("meta"):
        model = LanguageModel(**{k: config[k] for k in ("vocab_size", "dim", "n_layers", "n_heads", "hidden_dim", "max_seq_len")})
    # freqs_cis is deliberately not a checkpoint buffer in the Phase 2 model.
    model.freqs_cis = precompute_freqs_cis(config["dim"] // config["n_heads"], config["max_seq_len"] * 2)
    return model, config

def evaluate_loss(model, loader, device):
    model.eval(); losses = []
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            logits, _ = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100).item())
    model.train()
    return sum(losses) / max(1, len(losses))

def save(path, model, optimizer, scheduler, epoch, step, best_val, config):
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "step": step,
                "best_val_loss": best_val, "config": config}, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Print progress every N optimizer updates (default: 1).")
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.language}] Starting reasoning finetuning on {device}...", flush=True)
    model, base_config = build_model(args.language, device)
    language_dir = ROOT / args.language
    sp = spm.SentencePieceProcessor(model_file=str(language_dir / "tokenizer" / f"{args.language}_tokenizer.model"))
    data_dir = ROOT / "phase3" / "data" / args.language
    train_set = ReasoningDataset(data_dir / "train.jsonl", sp, base_config["max_seq_len"])
    val_set = ReasoningDataset(data_dir / "val.jsonl", sp, base_config["max_seq_len"])
    collator = lambda batch: collate(batch, sp.pad_id() if sp.pad_id() >= 0 else 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    start_epoch = step = 0; best_val = float("inf")
    checkpoint = args.resume or args.pretrained_checkpoint or language_dir / "checkpoints" / "final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint}. Download the Phase 2 final.pt first.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.pop("model_state_dict"), assign=True)
    # assign=True replaces Parameter objects independently, whereas the
    # original architecture intentionally ties input and output embeddings.
    # Restore the tie before creating/loading the optimizer so its parameter
    # groups exactly match the saved Phase 2/Phase 3 checkpoint.
    model.tok_embeddings.weight = model.output.weight
    model.to(device)
    print(f"[{args.language}] Loaded checkpoint: {checkpoint}", flush=True)
    # Create the optimizer only after assign=True has replaced the meta
    # parameters with checkpoint tensors; otherwise it would hold stale meta
    # parameter references.
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, updates_per_epoch * args.epochs), eta_min=args.lr * 0.1)
    if args.resume:
        optimizer.load_state_dict(state["optimizer_state_dict"]); scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch, step, best_val = state["epoch"] + 1, state["step"], state.get("best_val_loss", best_val)
    # Discard source optimizer/scheduler state before the training loop unless
    # it has just been loaded into the new fine-tuning optimizer.
    del state
    output = ROOT / "phase3" / "checkpoints" / args.language
    output.mkdir(parents=True, exist_ok=True)
    run_config = {**base_config, "phase": 3, "task": "reasoning_sft", "lr": args.lr, "epochs": args.epochs,
                  "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps, "seed": args.seed}
    log_path = output / "training_log.jsonl"
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    for epoch in range(start_epoch, args.epochs):
        print(f"[{args.language}] Epoch {epoch + 1}/{args.epochs} ({len(train_loader)} batches, {updates_per_epoch} optimizer updates)", flush=True)
        model.train(); optimizer.zero_grad(); running = 0.0
        for batch_idx, (x, labels) in enumerate(train_loader):
            x, labels = x.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda", dtype=torch.float16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
                scaled_loss = loss / args.grad_accum_steps
            if scaler: scaler.scale(scaled_loss).backward()
            else: scaled_loss.backward()
            running += loss.item()
            if (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx + 1 == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                scheduler.step(); optimizer.zero_grad(); step += 1
                if step % args.log_interval == 0:
                    print(f"[{args.language}] epoch={epoch + 1} update={step} batch={batch_idx + 1}/{len(train_loader)} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        val_loss = evaluate_loss(model, val_loader, device)
        record = {"epoch": epoch + 1, "step": step, "train_loss": running / len(train_loader), "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]}
        with log_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        save(output / "last.pt", model, optimizer, scheduler, epoch, step, best_val, run_config)
        if val_loss < best_val:
            best_val = val_loss; save(output / "best.pt", model, optimizer, scheduler, epoch, step, best_val, run_config)
        print(json.dumps(record), flush=True)
    print(f"Finished. Best checkpoint: {output / 'best.pt'}", flush=True)

if __name__ == "__main__": main()
