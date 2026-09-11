#!/usr/bin/env python3
"""Full Phase 2 Evaluation Suite for the Ablated Model (No Positional Embeddings).

Evaluates:
1. Intrinsic LM metrics: Cross-Entropy Loss, Perplexity (PPL), Bits-Per-Byte (BPB)
2. Generation metrics: BLEU-4, chrF, ROUGE-L across temperatures (0.0, 0.5, 1.0, 1.5)
3. Diversity & repetition diagnostics: Distinct-1, Distinct-2, Repetition Rate
4. Attention analysis: Layer 0 and Layer 5 heatmaps (Head 0, Head 2), Entropy, Distance
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
import seaborn as sns
import sentencepiece as spm
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bonus" / "scripts"))
sys.path.insert(0, str(ROOT / "hindi" / "scripts"))

from dataloader import get_dataloader
from model_no_pos import LanguageModelNoPos

def load_ablated_model(checkpoint_path, vocab_size=32000, device="cuda"):
    model = LanguageModelNoPos(vocab_size=vocab_size, max_seq_len=512)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def compute_intrinsic_metrics(model, dataloader, device, max_batches=100):
    print("Computing Intrinsic Language Modeling Metrics (PPL, BPB)...")
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Intrinsic Evaluation", total=max_batches):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
                _, loss = model(x, targets=y)
            total_loss += loss.item()
            total_steps += 1
            if total_steps >= max_batches:
                break
    
    avg_loss = total_loss / max(1, total_steps)
    ppl = math.exp(avg_loss)
    bpb = avg_loss / math.log(2)
    return {
        "cross_entropy_loss": round(avg_loss, 4),
        "perplexity": round(ppl, 4),
        "bits_per_byte": round(bpb, 4)
    }

def generate_continuation(model, sp, prefix, max_new_tokens=50, temp=1.0, device="cuda"):
    tokens = sp.encode_as_ids(prefix)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
                logits, _ = model(x)
            
            next_logits = logits[0, -1, :]
            if temp == 0.0:
                next_token = int(torch.argmax(next_logits, dim=-1).item())
            else:
                probs = F.softmax(next_logits / temp, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
                
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
            if x.shape[1] > 512:
                x = x[:, -512:]
            if next_token == sp.eos_id():
                break
                
    return sp.decode_ids(tokens + generated), sp.decode_ids(generated)

def evaluate_generation_and_diversity(model, sp, dataloader, device, num_samples=50):
    print("\nEvaluating Generation Metrics (BLEU-4, chrF, ROUGE-L, Distinct-1/2)...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    references = []
    hypotheses = []
    ref_ids_list = []
    hyp_ids_list = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            prefix = x[0, :10].unsqueeze(0)
            target_ids = y[0, 9:59].cpu().tolist()
            ref_text = sp.decode_ids(target_ids)
            
            current_x = prefix
            for _ in range(50):
                with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
                    logits, _ = model(current_x)
                next_token = int(torch.argmax(logits[0, -1, :], dim=-1).item())
                current_x = torch.cat([current_x, torch.tensor([[next_token]], device=device)], dim=1)
                if next_token == sp.eos_id():
                    break
                    
            gen_ids = current_x[0, 10:].cpu().tolist()
            gen_text = sp.decode_ids(gen_ids)
            
            references.append(ref_text)
            hypotheses.append(gen_text)
            ref_ids_list.append(" ".join(map(str, target_ids)))
            hyp_ids_list.append(" ".join(map(str, gen_ids)))
            
            if len(references) >= num_samples:
                break
                
    bleu = sacrebleu.corpus_bleu(hypotheses, [[r] for r in references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [[r] for r in references])
    rouge_l = sum(scorer.score(r, h)['rougeL'].fmeasure for r, h in zip(ref_ids_list, hyp_ids_list)) / max(1, len(references))
    
    unigrams, bigrams = set(), set()
    total_words, total_bigrams, repeated_bigrams = 0, 0, 0
    
    for h in hypotheses:
        words = h.split()
        total_words += len(words)
        seen = set()
        for i in range(len(words) - 1):
            bg = (words[i], words[i+1])
            bigrams.add(bg)
            total_bigrams += 1
            if bg in seen:
                repeated_bigrams += 1
            seen.add(bg)
        for w in words:
            unigrams.add(w)
            
    d1 = len(unigrams) / max(1, total_words)
    d2 = len(bigrams) / max(1, total_bigrams)
    rep_rate = repeated_bigrams / max(1, total_bigrams)
    
    return {
        "bleu_4": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "rouge_l": round(rouge_l, 4),
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "repetition_rate": round(rep_rate, 4),
        "sample_hypotheses": hypotheses[:3],
        "sample_references": references[:3]
    }

def analyze_attention_no_pos(model, sp, text, output_dirs, device="cuda"):
    print("\nRunning Attention Analysis on Ablated Model...")
    tokens = sp.encode_as_ids(text)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
            _, _, attentions = model(x, output_attentions=True)
            
    tokens_str = [sp.id_to_piece(t).replace(' ', ' ') for t in tokens]
    
    # Save heatmaps for Layer 0 and Layer 5, Heads 0 and 2
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        for layer_idx, head_idx in [(0, 0), (0, 2), (5, 0), (5, 2)]:
            attn_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn_matrix, xticklabels=tokens_str, yticklabels=tokens_str, cmap="viridis")
            plt.title(f"Ablation (No Pos): Layer {layer_idx} Head {head_idx}", fontname="DejaVu Sans")
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            out_file = os.path.join(out_dir, f"attn_L{layer_idx}_H{head_idx}_no_pos.png")
            plt.savefig(out_file, dpi=180)
            plt.close()
            print(f"Saved attention heatmap to {out_file}")

    # Compute entropy and mean attention distance across layers
    metrics_by_layer = {}
    for l, attn in enumerate(attentions):
        attn_probs = attn[0].float().cpu().numpy()  # (n_heads, seqlen, seqlen)
        
        # Entropy: -sum(p log p)
        entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-12), axis=-1)
        mean_entropy = float(np.mean(entropy))
        
        # Distance: sum(p * |i - j|)
        seq_len = attn_probs.shape[-1]
        distances = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])
        mean_dist = float(np.mean(np.sum(attn_probs * distances, axis=-1)))
        
        metrics_by_layer[f"layer_{l}"] = {
            "mean_entropy": round(mean_entropy, 4),
            "mean_distance": round(mean_dist, 4)
        }
        print(f"Layer {l}: Mean Entropy = {mean_entropy:.4f} | Mean Distance = {mean_dist:.4f}")

    return metrics_by_layer

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ablated Model (No Positional Embeddings)")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "bonus" / "checkpoints" / "best.pt"))
    parser.add_argument("--tokenizer", type=str, default=str(ROOT / "hindi" / "tokenizer" / "hindi_tokenizer.model"))
    parser.add_argument("--test-bin", type=str, default=str(ROOT / "hindi" / "data" / "hindi_test.bin"))
    parser.add_argument("--output-json", type=str, default=str(ROOT / "bonus" / "results" / "evaluation_metrics.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating Ablated Model on device: {device}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found. Please train the model first.")
        sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = load_ablated_model(args.checkpoint, vocab_size=32000, device=device)
    
    test_dl = get_dataloader(args.test_bin, batch_size=4, seq_len=512, shuffle=False)

    intrinsic = compute_intrinsic_metrics(model, test_dl, device=device, max_batches=100)
    print("\nIntrinsic Metrics:", json.dumps(intrinsic, indent=2))

    gen_metrics = evaluate_generation_and_diversity(model, sp, test_dl, device=device, num_samples=50)
    print("\nGeneration & Diversity Metrics:", json.dumps({k: v for k, v in gen_metrics.items() if not k.startswith("sample")}, indent=2))

    qualitative = {}
    prompts = ["भारत एक", "विज्ञान के क्षेत्र में", "हिमालय भारत का"]
    print("\n--- Qualitative Generations across Temperatures ---")
    for prompt in prompts:
        qualitative[prompt] = {}
        for temp in [0.0, 0.5, 1.0, 1.5]:
            full_text, continuation = generate_continuation(model, sp, prompt, max_new_tokens=40, temp=temp, device=device)
            qualitative[prompt][f"temp_{temp}"] = continuation
            print(f"[{prompt}] (Temp {temp}): {continuation}")

    attention_dirs = [
        str(ROOT / "bonus" / "images"),
        str(ROOT / "report" / "images" / "bonus")
    ]
    sample_sentence = "भारत एक बहुत ही सुंदर और विशाल देश है।"
    attn_metrics = analyze_attention_no_pos(model, sp, sample_sentence, attention_dirs, device=device)

    all_results = {
        "model": "Ablated Decoder-Only Transformer (No Positional Embeddings)",
        "language": "Hindi (Model H)",
        "intrinsic_metrics": intrinsic,
        "generation_metrics": gen_metrics,
        "qualitative_samples": qualitative,
        "attention_metrics": attn_metrics
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved complete evaluation results to {args.output_json}")

if __name__ == "__main__":
    main()
