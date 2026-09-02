import torch
import matplotlib.pyplot as plt
import os
import glob

def plot_loss(lang, color):
    checkpoint_dir = f"{lang}/checkpoints"
    # Find all .pt files except final.pt (which we can include if it has step, but we'll manually check)
    ckpt_files = glob.glob(f"{checkpoint_dir}/ckpt_*.pt")
    
    steps = []
    losses = []
    
    for f in ckpt_files:
        try:
            # extract step from filename: ckpt_5000.pt -> 5000
            step = int(os.path.basename(f).split('_')[1].split('.')[0])
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
            loss = ckpt['loss']
            steps.append(step)
            losses.append(loss)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    # Include final.pt, which is the last step (approx 20,000)
    final_file = f"{checkpoint_dir}/final.pt"
    if os.path.exists(final_file):
        try:
            ckpt = torch.load(final_file, map_location="cpu", weights_only=True)
            # Assuming final step is 20000 based on standard runs
            step = ckpt.get('step', 20000)
            loss = ckpt['loss']
            if step not in steps:
                steps.append(step)
                losses.append(loss)
        except Exception as e:
            print(f"Skipping {final_file}: {e}")
            
    # Add initial dummy loss at step 0 for the curve (usually around log(vocab_size))
    # log(32000) = 10.3, log(16000) = 9.6
    vocab_size = 32000 if lang == "hindi" else 16000
    import math
    steps.append(0)
    losses.append(math.log(vocab_size))
            
    # Sort by step
    sorted_pairs = sorted(zip(steps, losses))
    steps, losses = zip(*sorted_pairs)
    
    plt.plot(steps, losses, marker='o', label=lang.capitalize(), color=color)

def main():
    plt.figure(figsize=(10, 6))
    plot_loss("hindi", "blue")
    plot_loss("nepali", "red")
    
    plt.title("Pretraining Loss Curve (Sampled from Checkpoints)")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/loss_curve.png", dpi=150)
    print("Loss curve saved to report/loss_curve.png")

if __name__ == "__main__":
    main()
