import torch

class ModelConfig:
    """
    Settings for our GPT-style Transformer.
    You can tweak these numbers depending on your GPU memory.
    """
    # -------------------------------------------------------------------------
    # 1. Model Structure
    # -------------------------------------------------------------------------
    vocab_size = 50257        # Total number of unique tokens the model can understand
    n_embd = 512              # Size of each token’s embedding vector (smaller = uses less GPU memory)
    n_head = 8                # How many attention “heads” the model uses in each layer
    n_layer = 6               # Number of Transformer layers (more layers = bigger model)
    block_size = 256          # How many tokens the model looks back at once (context window)
    dropout = 0.1             # How much we randomly drop connections to prevent overfitting

    # -------------------------------------------------------------------------
    # 2. Training Settings
    # -------------------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, else CPU
    batch_size = 16           # Number of sequences processed at once (smaller = fits in 4GB GPU)
    learning_rate = 3e-4      # How fast the model updates weights; standard value for GPT training
    max_iters = 5000          # Total number of training steps
    eval_interval = 500       # Check performance every 500 steps

    # -------------------------------------------------------------------------
    # 3. File Paths
    # -------------------------------------------------------------------------
    model_path = "models/transformer_model.pth"  # Where the trained model will be saved

    def __init__(self):
        # Make sure the embedding size can be evenly split across attention heads
        assert self.n_embd % self.n_head == 0, "Embedding size must be divisible by number of heads"

# Create a global instance so other scripts can easily import these settings
cfg = ModelConfig()
