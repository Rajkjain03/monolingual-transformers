#!/usr/bin/env python3
"""Exact-match evaluation of pretrained and finetuned reasoning checkpoints."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[2]

def load(language, checkpoint, device):
    language_dir = ROOT / language
    sys.path.insert(0, str(language_dir / "scripts"))
    from model import LanguageModel, precompute_freqs_cis
    config = json.loads((language_dir / "configs" / "config.json").read_text())
    with torch.device("meta"):
        model = LanguageModel(**{k: config[k] for k in ("vocab_size", "dim", "n_layers", "n_heads", "hidden_dim", "max_seq_len")})
    model.freqs_cis = precompute_freqs_cis(config["dim"] // config["n_heads"], config["max_seq_len"] * 2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.pop("model_state_dict"), assign=True)
    model.tok_embeddings.weight = model.output.weight
    del state
    model.to(device); model.eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(language_dir / "tokenizer" / f"{language}_tokenizer.model"))
    return model, tokenizer, config["max_seq_len"]

@torch.no_grad()
def generate(model, sp, prompt, device, max_seq_len, max_new_tokens=12):
    ids = sp.encode_as_ids(prompt)
    generated = []
    for _ in range(max_new_tokens):
        x = torch.tensor(ids[-max_seq_len:], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        token = int(torch.argmax(logits[0, -1]).item())
        if token == sp.eos_id(): break
        generated.append(token); ids.append(token)
    return sp.decode_ids(generated).strip()

def predicted_entity(text, candidates):
    # An entity-name exact match is robust to harmless trailing punctuation.
    found = [name for name in candidates if name in text]
    return found[0] if len(found) == 1 else ""

def evaluate(model, sp, rows, device, max_seq_len, label, progress_interval):
    scored = []
    for index, row in enumerate(rows, start=1):
        completion = generate(model, sp, row["prompt"] + "\n" + row["answer_prefix"], device, max_seq_len)
        predicted = predicted_entity(completion, row["entities"])
        scored.append({"id": row["id"], "task_type": row["task_type"], "attribute": row["attribute"],
                       "prompt": row["prompt"], "expected": row["answer"], "completion": completion,
                       "predicted": predicted, "correct": predicted == row["answer"]})
        if index % progress_interval == 0 or index == len(rows):
            print(f"[{label}] evaluated {index}/{len(rows)} prompts", flush=True)
    return scored

def summary(rows):
    groups = defaultdict(list)
    for row in rows: groups[row["task_type"]].append(row["correct"])
    groups["overall"] = [row["correct"] for row in rows]
    return {name: {"correct": sum(values), "total": len(values), "accuracy": round(sum(values) / len(values), 4)} for name, values in groups.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--finetuned-checkpoint", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained = args.pretrained_checkpoint or ROOT / args.language / "checkpoints" / "final.pt"
    finetuned = args.finetuned_checkpoint or ROOT / "phase3" / "checkpoints" / args.language / "best.pt"
    test_file = args.test_file or ROOT / "phase3" / "data" / args.language / "test.jsonl"
    for path in (pretrained, finetuned, test_file):
        if not path.exists(): raise FileNotFoundError(path)
    rows = [json.loads(line) for line in test_file.read_text(encoding="utf-8").splitlines()]
    print(f"[{args.language}] Loading pretrained checkpoint on {device}...", flush=True)
    pre_model, sp, max_len = load(args.language, pretrained, device)
    pre_rows = evaluate(pre_model, sp, rows, device, max_len, f"{args.language} pretrained", args.progress_interval)
    print(f"[{args.language}] Loading finetuned checkpoint...", flush=True)
    ft_model, sp, max_len = load(args.language, finetuned, device)
    ft_rows = evaluate(ft_model, sp, rows, device, max_len, f"{args.language} finetuned", args.progress_interval)
    fixed = next(({"pretrained": p, "finetuned": f} for p, f in zip(pre_rows, ft_rows) if not p["correct"] and f["correct"]), None)
    failure = next((f for f in ft_rows if not f["correct"] and f["task_type"] == "transitive"), next((f for f in ft_rows if not f["correct"]), None))
    output = ROOT / "phase3" / "results"; output.mkdir(parents=True, exist_ok=True)
    report = {"language": args.language, "metric": "entity exact-match accuracy", "test_file": str(test_file),
              "pretrained": summary(pre_rows), "finetuned": summary(ft_rows), "fixed_example": fixed, "failure_example": failure}
    (output / f"{args.language}_reasoning_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / f"{args.language}_pretrained_predictions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pre_rows), encoding="utf-8")
    (output / f"{args.language}_finetuned_predictions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ft_rows), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
