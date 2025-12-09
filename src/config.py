import torch
import os

class ModelConfig:
    """
    Settings for our GPT-style Transformer.
    Can tweak depending on GPU memory and training needs.
    """
    # -------------------------------------------------------------------------
    # 1. Model Structure
    # -------------------------------------------------------------------------
    vocab_size = 50257        # Total number of unique tokens
    n_embd = 768              # Embedding size
    n_head = 12               # Attention heads
    n_layer = 12              # Transformer layers
    block_size = 1024         # Context window
    dropout = 0.1             # Dropout rate

    # -------------------------------------------------------------------------
    # 2. Training Settings
    # -------------------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    learning_rate = 3e-4
    max_iters = 10000
    eval_interval = 500
    accumulation_steps = 4    # For gradient accumulation

    # -------------------------------------------------------------------------
    # 3. File Paths
    # -------------------------------------------------------------------------
    model_save_dir = "GPT-Implementation/models"  # Folder for all checkpoints
    model_path = os.path.join(model_save_dir, "model_finalV2.pth")             # Final model

    def __init__(self):
        # Ensure embedding can be split across attention heads
        assert self.n_embd % self.n_head == 0, "Embedding size must be divisible by number of heads"
        # Make sure the save folder exists
        os.makedirs(self.model_save_dir, exist_ok=True)
        print(f"Model checkpoints will be saved to: {self.model_save_dir}")

# Global instance for easy import
cfg = ModelConfig()
