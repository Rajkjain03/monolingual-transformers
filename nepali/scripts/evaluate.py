import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import sentencepiece as spm
import sacrebleu
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataloader import get_dataloader
from model import LanguageModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_path, spm_path, vocab_size):
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    model = LanguageModel(vocab_size=vocab_size, max_seq_len=512)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model, sp

def compute_intrinsic_metrics(model, dataloader):
    total_loss = 0
    total_steps = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Intrinsic Metrics"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast(DEVICE, dtype=torch.float16):
                logits, loss = model(x, targets=y)
            total_loss += loss.item()
            total_steps += 1
            if total_steps >= 500: # Evaluate on 500 batches
                break
    
    avg_loss = total_loss / total_steps
    ppl = math.exp(avg_loss)
    bpb = avg_loss / math.log(2)
    return avg_loss, ppl, bpb

def generate(model, sp, prefix, max_new_tokens=50, temp=1.0):
    tokens = sp.encode_as_ids(prefix)
    x = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.amp.autocast(DEVICE, dtype=torch.float16):
                logits, _ = model(x)
            
            next_token_logits = logits[0, -1, :]
            
            if temp == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1).item()
            else:
                next_token_logits = next_token_logits / temp
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]], device=DEVICE)], dim=1)
            
            if x.shape[1] > 512:
                x = x[:, -512:]
                
            if next_token == sp.eos_id():
                break
                
    return sp.decode_ids(tokens + generated)

def evaluate_generation(model, sp):
    # Some basic prompts for testing Nepali
    prompts = [
        "नेपाल एक",
        "विज्ञानको क्षेत्रमा",
        "उहाँले भन्नुभयो कि"
    ]
    
    print("\n--- Generation Quality ---")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        for temp in [0.5, 1.0, 1.5]:
            gen = generate(model, sp, prompt, temp=temp)
            print(f"[Temp {temp}]: {gen}")

def attention_analysis(model, sp, text, out_dir):
    tokens = sp.encode_as_ids(text)
    x = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        with torch.amp.autocast(DEVICE, dtype=torch.float16):
            _, _, attentions = model(x, output_attentions=True)
            
    # attentions is list of (1, n_heads, seq_len, seq_len)
    tokens_str = [sp.id_to_piece(t).replace(' ', ' ') for t in tokens]
    
    os.makedirs(out_dir, exist_ok=True)
    
    def plot_heatmap(attn_matrix, layer_idx, head_idx):
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_matrix, xticklabels=tokens_str, yticklabels=tokens_str, cmap="viridis")
        plt.title(f"Layer {layer_idx} Head {head_idx}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/attn_L{layer_idx}_H{head_idx}.png")
        plt.close()
        
    # Plot Layer 0 (Early) Head 0
    attn_l0_h0 = attentions[0][0, 0].cpu().numpy()
    plot_heatmap(attn_l0_h0, 0, 0)
    
    # Plot Layer 5 (Late) Head 0
    attn_l5_h0 = attentions[-1][0, 0].cpu().numpy()
    plot_heatmap(attn_l5_h0, 5, 0)
    
    # Calculate Mean Attention Distance and Entropy
    print("\n--- Attention Analysis ---")
    for l, attn in enumerate(attentions):
        attn_probs = attn[0].float().cpu().numpy() # (heads, seq_len, seq_len)
        
        # Entropy: -sum(p log p)
        entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-12), axis=-1)
        mean_entropy = np.mean(entropy)
        
        # Distance: sum(p * |i - j|)
        seq_len = attn_probs.shape[-1]
        distances = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])
        mean_dist = np.mean(np.sum(attn_probs * distances, axis=-1))
        
        print(f"Layer {l}: Mean Entropy = {mean_entropy:.4f} | Mean Distance = {mean_dist:.4f}")

if __name__ == "__main__":
    print("Evaluating Nepali Model...")
    model_path = "nepali/checkpoints/final.pt"
    spm_path = "nepali/tokenizer/nepali_tokenizer.model"
    
    model, sp = load_model_and_tokenizer(model_path, spm_path, vocab_size=16000)
    
    val_dl = get_dataloader("nepali/data/nepali_test.bin", batch_size=4, seq_len=512, shuffle=False)
    
    loss, ppl, bpb = compute_intrinsic_metrics(model, val_dl)
    print(f"\nIntrinsic Metrics:")
    print(f"Cross-Entropy Loss: {loss:.4f}")
    print(f"Perplexity (PPL): {ppl:.4f}")
    print(f"Bits-Per-Byte (BPB): {bpb:.4f}")
    
    evaluate_generation(model, sp)
    attention_analysis(model, sp, "नेपाल एक धेरै सुन्दर र विशाल देश हो।", "nepali/eval_plots")
