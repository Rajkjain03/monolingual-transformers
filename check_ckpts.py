import torch
import glob
for lang in ['hindi', 'nepali']:
    print(f"\n{lang.upper()}:")
    for f in sorted(glob.glob(f"{lang}/checkpoints/*.pt")):
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
        step = ckpt.get('step', 'unknown')
        loss = ckpt.get('loss', 'unknown')
        print(f"File: {f} | Step: {step} | Loss: {loss}")
