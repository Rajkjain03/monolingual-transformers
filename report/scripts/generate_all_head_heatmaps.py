import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import sentencepiece as spm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from hindi.scripts.model import LanguageModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {DEVICE}")

# Set default font to DejaVu Sans for English/labels
plt.rcParams['font.family'] = 'DejaVu Sans'
devanagari_fp = fm.FontProperties(family='Noto Sans Devanagari')

def classify_head(entropy, distance):
    if distance < 1.2:
        if entropy < 0.4:
            return "Hyper-local n-gram (sharp)"
        return "Local n-gram extractor"
    elif distance < 2.2:
        if entropy > 1.1:
            return "Broad local / diffuse syntax"
        return "Local / phrase-level syntax"
    elif distance < 3.0:
        if entropy > 1.2:
            return "Diffuse multi-token routing"
        return "Mid-range relational binding"
    else:
        if entropy < 0.3:
            return "Extreme Attention Sink (Token 0 offload)"
        elif entropy < 0.6:
            return "Strong Attention Sink / Punctuation sink"
        elif entropy < 1.0:
            return "Long-range semantic content"
        else:
            return "Broad global context"

def process_language(lang, model_path, sp_path, vocab_size, benchmark_text):
    print(f"\n==========================================")
    print(f"Processing {lang.upper()} (All Layers & All Heads)")
    print(f"==========================================")
    
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    tokens = sp.encode_as_ids(benchmark_text)
    clean_tokens = [sp.id_to_piece(t).replace('\u2581', '').replace(' ', '').strip() for t in tokens]
    token_labels = [t if t != '' else '_' for t in clean_tokens]
    raw_pieces = [sp.id_to_piece(t) for t in tokens]
    T = len(tokens)
    print(f"Benchmark sentence: '{benchmark_text}'")
    print(f"Token count: {T} tokens: {token_labels}")
    
    model = LanguageModel(vocab_size=vocab_size, max_seq_len=512)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, _, all_attentions = model(x, output_attentions=True)
        
    metrics_data = {
        "language": lang,
        "benchmark_text": benchmark_text,
        "token_count": T,
        "tokens": raw_pieces,
        "clean_tokens": token_labels,
        "layers": {}
    }
    
    dist_matrix = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    
    os.makedirs("report/images/all_heads", exist_ok=True)
    
    # 1. Compute quantitative metrics and save individual separate head heatmaps
    for l_idx in range(6):
        metrics_data["layers"][l_idx] = {}
        layer_attn = all_attentions[l_idx][0].float().cpu().numpy() # (8, T, T)
        for h_idx in range(8):
            head_probs = layer_attn[h_idx] # (T, T)
            entropy = float(-np.sum(head_probs * np.log(head_probs + 1e-12), axis=-1).mean())
            mean_dist = float(np.sum(head_probs * dist_matrix, axis=-1).mean())
            role = classify_head(entropy, mean_dist)
            
            metrics_data["layers"][l_idx][h_idx] = {
                "entropy": round(entropy, 4),
                "distance": round(mean_dist, 4),
                "role": role
            }
            
            # Save high-res individual separate heatmap with explicit X and Y axis coordinate descriptions
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(head_probs, cmap="viridis", vmin=0, vmax=1.0, aspect="auto")
            
            # Colorbar with label
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Attention Weight $A_{ij} \in [0, 1]$", fontsize=10)
            
            # Ticks
            ax.set_xticks(np.arange(T))
            ax.set_yticks(np.arange(T))
            ax.set_xticklabels(token_labels, fontproperties=devanagari_fp, rotation=45, ha="right", fontsize=11)
            ax.set_yticklabels(token_labels, fontproperties=devanagari_fp, fontsize=11)
            
            # Explicit coordinate definitions on X and Y axes
            ax.set_xlabel("X-Axis: Key Token $K_j$ (Attended Context Token $j$)", fontsize=11, fontweight="bold", labelpad=8)
            ax.set_ylabel("Y-Axis: Query Token $Q_i$ (Attending Token $i$)", fontsize=11, fontweight="bold", labelpad=8)
            
            ax.set_title(
                f"{lang.capitalize()} Layer {l_idx} Head {h_idx}: {role}\n"
                f"(Shannon Entropy = {entropy:.4f}, Mean Distance = {mean_dist:.4f})",
                fontsize=12, fontweight="bold", pad=12
            )
            
            # Add gridlines between cells
            ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, T, 1), minor=True)
            ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.tick_params(which="minor", bottom=False, left=False)
            
            plt.tight_layout()
            plt.savefig(f"report/images/all_heads/{lang}_L{l_idx}_H{h_idx}.png", dpi=200)
            plt.close()
            
    with open(f"report/images/all_heads_{lang}_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics JSON: report/images/all_heads_{lang}_metrics.json")
    
    # 2. Master 6x8 Grid Heatmap with explicit coordinate labels
    fig, axes = plt.subplots(6, 8, figsize=(24, 18), constrained_layout=True)
    for l_idx in range(6):
        layer_attn = all_attentions[l_idx][0].float().cpu().numpy()
        for h_idx in range(8):
            ax = axes[l_idx, h_idx]
            im = ax.imshow(layer_attn[h_idx], cmap="viridis", vmin=0, vmax=1.0, aspect="auto")
            m = metrics_data["layers"][l_idx][h_idx]
            ax.set_title(f"L{l_idx} H{h_idx} (D={m['distance']:.1f}, H={m['entropy']:.2f})", fontsize=8.5, fontweight="bold")
            
            if l_idx == 5:
                ax.set_xticks(np.arange(T))
                ax.set_xticklabels(token_labels, fontproperties=devanagari_fp, rotation=90, fontsize=8)
            else:
                ax.set_xticks([])
                
            if h_idx == 0:
                ax.set_yticks(np.arange(T))
                ax.set_yticklabels(token_labels, fontproperties=devanagari_fp, fontsize=8)
            else:
                ax.set_yticks([])
                
    # Big overarching axis titles explaining coordinate system
    fig.suptitle(
        f"{lang.upper()} Transformer Decoder: Complete 48-Head Attention Hierarchy (6 Layers × 8 Heads)\n"
        f"Cell (Row i, Col j): Attention probability from Query Token i (Y-axis) to Key Token j (X-axis)",
        fontsize=16, fontweight="bold", y=1.02
    )
    fig.supxlabel("X-Axis: Key Tokens $K_j$ [Columns: Context tokens being attended to]", fontsize=14, fontweight="bold")
    fig.supylabel("Y-Axis: Query Tokens $Q_i$ [Rows: Current generating tokens]", fontsize=14, fontweight="bold")
    
    grid_path = f"report/images/all_layers_all_heads_{lang}.png"
    plt.savefig(grid_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved master 6x8 grid heatmap: {grid_path}")
    
    # 3. Save Per-layer 1x8 mosaics with explicit X and Y axis coordinates
    for l_idx in range(6):
        fig, axes = plt.subplots(1, 8, figsize=(24, 3.5), constrained_layout=True)
        layer_attn = all_attentions[l_idx][0].float().cpu().numpy()
        for h_idx in range(8):
            ax = axes[h_idx]
            ax.imshow(layer_attn[h_idx], cmap="viridis", vmin=0, vmax=1.0, aspect="auto")
            m = metrics_data["layers"][l_idx][h_idx]
            ax.set_title(f"Head {h_idx}\nD={m['distance']:.2f}, H={m['entropy']:.2f}", fontsize=10, fontweight="bold")
            ax.set_xticks(np.arange(T))
            ax.set_xticklabels(token_labels, fontproperties=devanagari_fp, rotation=90, fontsize=8.5)
            if h_idx == 0:
                ax.set_yticks(np.arange(T))
                ax.set_yticklabels(token_labels, fontproperties=devanagari_fp, fontsize=8.5)
                ax.set_ylabel("Query Token $Q_i$ [Row]", fontsize=10, fontweight="bold")
            else:
                ax.set_yticks([])
            if h_idx == 3:
                ax.set_xlabel("Key Token $K_j$ [Column: Attended Context]", fontsize=10, fontweight="bold")
                
        fig.suptitle(f"{lang.upper()} Layer {l_idx} (All 8 Heads) | Rows = Query $Q_i$, Columns = Key $K_j$", fontsize=12, fontweight="bold")
        mosaic_path = f"report/images/all_heads/{lang}_layer_{l_idx}_all_heads.png"
        plt.savefig(mosaic_path, dpi=200, bbox_inches="tight")
        plt.close()
    print(f"Saved per-layer 1x8 mosaics for {lang}")
    
    # 4. Generate a 4-archetype separate detailed comparison image for the report
    # Choosing archetypes: Local N-Gram, Syntactic Binding, Long-Range Semantic, Attention Sink
    if lang == "hindi":
        sample_heads = [
            (0, 1, "Archetype 1: Hyper-Local N-Gram"),
            (1, 2, "Archetype 2: Phrase / Relational Syntax"),
            (5, 0, "Archetype 3: Long-Range Semantic Content"),
            (5, 5, "Archetype 4: Attention Sink Offload (Token 0)")
        ]
    else:
        sample_heads = [
            (0, 7, "Archetype 1: Hyper-Local N-Gram"),
            (2, 3, "Archetype 2: Phrase / Relational Syntax"),
            (5, 0, "Archetype 3: Long-Range Semantic Content"),
            (5, 2, "Archetype 4: Attention Sink Offload (Token 0)")
        ]
        
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
    for idx, (l_idx, h_idx, arch_title) in enumerate(sample_heads):
        ax = axes[idx]
        attn = all_attentions[l_idx][0, h_idx].float().cpu().numpy()
        im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1.0, aspect="auto")
        m = metrics_data["layers"][l_idx][h_idx]
        
        ax.set_title(
            f"{arch_title}\nLayer {l_idx} Head {h_idx}\n(Dist={m['distance']:.2f}, Entropy={m['entropy']:.2f})",
            fontsize=11, fontweight="bold", pad=8
        )
        ax.set_xticks(np.arange(T))
        ax.set_yticks(np.arange(T))
        ax.set_xticklabels(token_labels, fontproperties=devanagari_fp, rotation=45, ha="right", fontsize=9.5)
        ax.set_yticklabels(token_labels, fontproperties=devanagari_fp, fontsize=9.5)
        
        ax.set_xlabel("X: Key Token $K_j$ (Attended Context)", fontsize=10, fontweight="bold", labelpad=6)
        if idx == 0:
            ax.set_ylabel("Y: Query Token $Q_i$ (Attending Token)", fontsize=10, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")
            
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight")
        
    fig.suptitle(
        f"{lang.upper()}: Four Core Functional Attention Archetypes\n"
        f"X-Axis = Key Context ($j$), Y-Axis = Query Position ($i$), Upper Triangle = Causal Mask (0.0)",
        fontsize=13, fontweight="bold", y=1.03
    )
    arch_path = f"report/images/separate_heads_{lang}_archetypes.png"
    plt.savefig(arch_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved separate archetype comparison: {arch_path}")

def main():
    hindi_benchmark = "भारत एक बहुत ही सुंदर और विशाल देश है।"
    nepali_benchmark = "नेपाल एक धेरै सुन्दर र विशाल देश हो।"
    
    process_language(
        lang="hindi",
        model_path="hindi/checkpoints/final.pt",
        sp_path="hindi/tokenizer/hindi_tokenizer.model",
        vocab_size=32000,
        benchmark_text=hindi_benchmark
    )
    
    process_language(
        lang="nepali",
        model_path="nepali/checkpoints/final.pt",
        sp_path="nepali/tokenizer/nepali_tokenizer.model",
        vocab_size=16000,
        benchmark_text=nepali_benchmark
    )

if __name__ == "__main__":
    main()
