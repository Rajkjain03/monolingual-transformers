import torch
import numpy as np
import sentencepiece as spm
from hindi.scripts.model import LanguageModel

def get_metrics(model_path, spm_path, text, vocab_size):
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    model = LanguageModel(vocab_size=vocab_size, max_seq_len=512)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokens = sp.encode_as_ids(text)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            _, _, attentions = model(x, output_attentions=True)
            
    seq_len = len(tokens)
    distances = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])
    
    metrics = {}
    for layer in [0, 5]:
        for head in [0, 2]:
            attn_probs = attentions[layer][0, head].float().cpu().numpy()
            entropy = np.mean(-np.sum(attn_probs * np.log(attn_probs + 1e-12), axis=-1))
            mean_dist = np.mean(np.sum(attn_probs * distances, axis=-1))
            metrics[f"L{layer}_H{head}"] = (entropy, mean_dist)
            
    return metrics

print("Hindi:")
h_metrics = get_metrics("hindi/checkpoints/final.pt", "hindi/tokenizer/hindi_tokenizer.model", "भारत एक बहुत ही सुंदर और विशाल देश है।", 32000)
for k, v in h_metrics.items(): print(f"{k}: Entropy={v[0]:.4f}, Dist={v[1]:.4f}")

print("\nNepali:")
n_metrics = get_metrics("nepali/checkpoints/final.pt", "nepali/tokenizer/nepali_tokenizer.model", "नेपाल एक धेरै सुन्दर र विशाल देश हो।", 16000)
for k, v in n_metrics.items(): print(f"{k}: Entropy={v[0]:.4f}, Dist={v[1]:.4f}")
