import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import cfg # Import the config object we made

class CausalSelfAttention(nn.Module):
    """
    The "Brain" of the Transformer.
    It allows tokens to look at previous tokens to understand context.
    """
    def __init__(self):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        
        # Key, Query, Value projections combined (3 * n_embd) for efficiency
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        
        # Output projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        
        # Causal Mask: Ensures the model cannot "cheat" by looking at future tokens
        # We use register_buffer so it's not treated as a trainable parameter
        bias = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("bias", bias.view(1, 1, cfg.block_size, cfg.block_size))
        
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        B, T, C = x.size() # Batch, Time (Sequence Length), Channels (Embed Dim)
        
        # 1. Calculate Query, Key, Value
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 2. Reshape for Multi-Head Attention
        # Transform from (B, T, C) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. Scaled Dot-Product Attention (The Math)
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 4. Apply Mask (The "Causal" part)
        # Fill future positions with -infinity so Softmax turns them to 0
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # 5. Softmax & Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        self.last_att_scores = att # We save the scores to read them later
        
        # 6. Aggregate Values
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # 7. Reassemble Heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    """
    The Feed-Forward Network.
    Process the information gathered by the attention heads.
    """
    def __init__(self):
        super().__init__()
        self.c_fc    = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu    = nn.GELU() # Newer activation function used in GPT
        self.c_proj  = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    A single Transformer Block: Communication (Attention) + Computation (MLP)
    """
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP()

    def forward(self, x):
        # Residual connections (x + ...) are crucial for deep networks
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """
    The Main Model Class.
    """
    def __init__(self):
        super().__init__()
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd), # Token Embeddings
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd), # Positional Embeddings
            drop = nn.Dropout(cfg.dropout),
            h = nn.ModuleList([Block() for _ in range(cfg.n_layer)]), # The Layers
            ln_f = nn.LayerNorm(cfg.n_embd), # Final normalization
        ))
        
        # The Final Head: Projects embeddings back to vocabulary size
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        # Weight initialization (Important for convergence)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # 1. Create Position Indices (0, 1, 2, ..., T-1)
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device) 
        
        # 2. Token + Positional Embeddings
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 3. Run through Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
            
        # 4. Final Norm
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # Training Mode: Calculate Loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            # Generation Mode (Inference)
            # Optimization: Only calculate logits for the very last position
            logits = self.lm_head(x[:, [-1], :]) 
            return logits, None