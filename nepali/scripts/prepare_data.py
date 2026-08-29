import os
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

def prepare_split(split, input_file, output_file, sp_model_path):
    print(f"Preparing {split} split...")
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    
    # First pass: calculate total tokens
    # To keep RAM usage incredibly low, we write in chunks to the binary file directly.
    # np.memmap requires a known size if we create it, but we can just stream append to a normal file
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    total_tokens = 0
    chunk_size = 100_000 # lines per chunk
    
    buffer = []
    
    with open(input_file, "r", encoding="utf-8") as f, open(output_file, "wb") as out_f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if not line:
                continue
            
            ids = sp.encode_as_ids(line)
            # Add EOS token manually if desired (SentencePiece doesn't auto-add it unless configured)
            # Let's add EOS (id 2) to separate documents
            ids.append(sp.eos_id())
            buffer.extend(ids)
            
            if len(buffer) > 10_000_000:
                # write buffer
                arr = np.array(buffer, dtype=np.uint16)
                out_f.write(arr.tobytes())
                total_tokens += len(buffer)
                buffer = []
                
        # write remainder
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            out_f.write(arr.tobytes())
            total_tokens += len(buffer)
            
    print(f"Saved {total_tokens:,} tokens to {output_file}")

if __name__ == "__main__":
    sp_path = "nepali/tokenizer/nepali_tokenizer.model"
    
    # Process splits
    splits = [
        ("train", "nepali/data/nepali_train.txt", "nepali/data/nepali_train.bin"),
        ("val", "nepali/data/nepali_val.txt", "nepali/data/nepali_val.bin"),
        ("test", "nepali/data/nepali_test.txt", "nepali/data/nepali_test.bin")
    ]
    
    for split_name, in_file, out_file in splits:
        if os.path.exists(in_file):
            prepare_split(split_name, in_file, out_file, sp_path)
        else:
            print(f"Skipping {split_name}, {in_file} not found.")
