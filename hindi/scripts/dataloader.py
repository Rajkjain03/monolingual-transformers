import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class MemmapDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len
        self.bin_path = bin_path
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Data file not found: {bin_path}")
            
        # Memory map the binary file
        # dtype is uint16 because our vocab size is < 65535 (32K for Hindi, 16K for Nepali)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
        # Calculate how many full sequences we can extract
        # We need seq_len + 1 tokens for each sample (x and y)
        self.num_samples = len(self.data) // (seq_len + 1)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * (self.seq_len + 1)
        end_idx = start_idx + self.seq_len + 1
        
        chunk = self.data[start_idx:end_idx].astype(np.int64)
        
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        
        return x, y

def get_dataloader(bin_path: str, batch_size: int, seq_len: int, shuffle: bool = True):
    dataset = MemmapDataset(bin_path, seq_len)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=2, 
        pin_memory=True
    )

if __name__ == "__main__":
    # Test dataloader
    dl = get_dataloader("hindi/data/hindi_val.bin", batch_size=4, seq_len=128)
    for x, y in dl:
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        break
