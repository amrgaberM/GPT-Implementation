import torch
from torch.utils.data import Dataset
import tiktoken
import numpy as np
import os

class TextDataset(Dataset):
    """
    A PyTorch Dataset that loads text, encodes it with GPT-2 tokenizer,
    and produces sliding window chunks for training.
    """
    def __init__(self, file_path, block_size):
        self.block_size = block_size
        
        # 1. Load the Text
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 2. Tokenize (Turn text into integers)
        # We use 'gpt2' encoding which is standard BPE (Byte Pair Encoding)
        enc = tiktoken.get_encoding("gpt2")
        self.data = np.array(enc.encode(text), dtype=np.int64)
        
        print(f"Loaded {len(self.data)} tokens from {file_path}")

    def __len__(self):
        # We can extract len(data) - block_size chunks
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # 3. Create Inputs (x) and Targets (y)
        # If text is "Hello how are you", block_size=3
        # x = "Hello how are"
        # y = "how are you" (The next word for every position)
        
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # Convert to PyTorch tensors
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        
        # NOTE: We return LongTensor (integers) because that's what Embedding layers expect
        return x, y