#!/usr/bin/env python3
"""Paired Phase 2-style attention heatmaps for a reasoning prompt."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams["font.family"] = "Noto Sans Devanagari"

def load(language, checkpoint, device):
    language_dir = ROOT / language; sys.path.insert(0, str(language_dir / "scripts"))
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
    return model, spm.SentencePieceProcessor(model_file=str(language_dir / "tokenizer" / f"{language}_tokenizer.model"))

@torch.no_grad()
def get_attentions(model, sp, text, device):
    ids = sp.encode_as_ids(text); x = torch.tensor(ids, device=device).unsqueeze(0)
    _, _, attention = model(x, output_attentions=True)
    return [a[0].float().cpu().numpy() for a in attention], [sp.id_to_piece(i) for i in ids]

def metrics(matrix):
    entropy = -np.sum(matrix * np.log(matrix + 1e-12), axis=-1).mean(axis=-1)
    positions = np.arange(matrix.shape[-1]); distances = np.abs(positions[:, None] - positions[None, :])
    distance = np.sum(matrix * distances, axis=-1).mean(axis=-1)
    return entropy, distance

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--language", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--pretrained-checkpoint", type=Path); parser.add_argument("--finetuned-checkpoint", type=Path)
    parser.add_argument("--prompt", help="Defaults to the first generated test prompt.")
    args = parser.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    pre = args.pretrained_checkpoint or ROOT / args.language / "checkpoints" / "final.pt"
    ft = args.finetuned_checkpoint or ROOT / "phase3" / "checkpoints" / args.language / "best.pt"
    for path in (pre, ft):
        if not path.exists(): raise FileNotFoundError(path)
    if args.prompt: prompt = args.prompt
    else: prompt = json.loads((ROOT / "phase3" / "data" / args.language / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])["prompt"]
    pre_model, sp = load(args.language, pre, device); ft_model, _ = load(args.language, ft, device)
    pre_attn, pieces = get_attentions(pre_model, sp, prompt, device); ft_attn, _ = get_attentions(ft_model, sp, prompt, device)
    output = ROOT / "report" / "images" / "phase3" / args.language; output.mkdir(parents=True, exist_ok=True)
    records = {"language": args.language, "prompt": prompt, "layers": {}}
    for layer in (0, 5):
        records["layers"][str(layer)] = {}
        for head in (0, 2):
            fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
            for ax, matrix, title in zip(axes, (pre_attn[layer][head], ft_attn[layer][head]), ("Pretrained", "Finetuned")):
                sns.heatmap(matrix, xticklabels=pieces, yticklabels=pieces, cmap="viridis", ax=ax)
                ax.set_title(f"{title}: L{layer} H{head}"); ax.tick_params(axis="x", rotation=90)
            fig.tight_layout(); fig.savefig(output / f"L{layer}_H{head}_pretrained_vs_finetuned.png", dpi=180); plt.close(fig)
            pe, pd = metrics(pre_attn[layer][head][None, ...]); fe, fd = metrics(ft_attn[layer][head][None, ...])
            records["layers"][str(layer)][str(head)] = {"pretrained": {"mean_entropy": float(pe[0]), "mean_distance": float(pd[0])}, "finetuned": {"mean_entropy": float(fe[0]), "mean_distance": float(fd[0])}}
    (output / "attention_metrics.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
