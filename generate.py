import torch
import tiktoken
from src.model import GPT
from src.config import cfg

def generate_text():
    # 1. Setup
    device = cfg.device
    print(f"Loading model on {device}...")
    
    # 2. Load the trained model
    model = GPT().to(device)
    # We use strict=False because we might have extra keys in the state_dict depending on saving
    try:
        model.load_state_dict(torch.load(cfg.model_path, map_location=device))
        print("Model weights loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find model at {cfg.model_path}. Did you finish training?")
        return

    model.eval() # Switch to evaluation mode (turns off Dropout)

    # 3. Prompt the model
    start_text = "The king said"
    print(f"\nPrompt: '{start_text}'")
    
    # Encode prompt
    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0) # Add batch dim (1, T)

    # 4. Generate
    print("Generating...")
    # We generate 100 tokens
    max_new_tokens = 100 
    
    with torch.no_grad(): # No need to calculate gradients for generation
        for _ in range(max_new_tokens):
            # Crop context if it gets too long (block_size)
            x_cond = x[:, -cfg.block_size:]
            
            # Get logits
            logits, _ = model(x_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] 
            
            # Apply Softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample from the distribution (makes it creative)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            x = torch.cat((x, idx_next), dim=1)

    # 5. Decode and Print
    generated_text = enc.decode(x[0].tolist())
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    generate_text()