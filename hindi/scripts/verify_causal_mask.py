import torch
import sentencepiece as spm
from model import LanguageModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def verify_causal_mask():
    print("Loading Hindi Model...")
    sp = spm.SentencePieceProcessor(model_file="hindi/tokenizer/hindi_tokenizer.model")
    model = LanguageModel(vocab_size=32000, max_seq_len=512)
    checkpoint = torch.load("hindi/checkpoints/final.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print("\n--- Causal Mask Empirical Verification ---")
    
    # Original sequence: "The quick brown fox" -> Let's use Hindi
    # "भारत एक बहुत ही सुंदर देश है"
    seq_A = "भारत एक बहुत ही सुंदर"
    # Changed sequence: change "सुंदर" to "विशाल"
    seq_B = "भारत एक बहुत ही विशाल"
    
    tokens_A = sp.encode_as_ids(seq_A)
    tokens_B = sp.encode_as_ids(seq_B)
    
    print(f"Sequence A: {sp.decode_ids(tokens_A)} -> {tokens_A}")
    print(f"Sequence B: {sp.decode_ids(tokens_B)} -> {tokens_B}")
    
    x_a = torch.tensor([tokens_A], dtype=torch.long, device=DEVICE)
    x_b = torch.tensor([tokens_B], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        with torch.amp.autocast(DEVICE, dtype=torch.float16):
            logits_A, _ = model(x_a)
            logits_B, _ = model(x_b)
            
    # Check logits at position 2 (index 2 corresponds to predicting token 3)
    # Token 3 is "बहुत" in both, but token 4 differs.
    # If the causal mask works, the logits predicting token 3 (which only depends on tokens 0..2)
    # must be exactly identical between A and B, despite token 4 being different.
    
    diff_pos_2 = torch.max(torch.abs(logits_A[0, 2, :] - logits_B[0, 2, :])).item()
    diff_pos_4 = torch.max(torch.abs(logits_A[0, 4, :] - logits_B[0, 4, :])).item()
    
    print(f"\nMax logit difference at position t=2 (before change): {diff_pos_2}")
    print(f"Max logit difference at position t=4 (at/after change): {diff_pos_4}")
    
    if diff_pos_2 == 0.0:
        print("\nVERIFICATION SUCCESSFUL: Changing token t+1 did NOT change logits at position t.")
        print("The model cannot see the future. The causal mask is working correctly.")
    else:
        print("\nVERIFICATION FAILED: Logits leaked from the future!")

if __name__ == "__main__":
    verify_causal_mask()
