import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.model import GPT
from src.dataset import TextDataset
from src.config import cfg
from tqdm import tqdm # Import the progress bar library
import os

# Ensure the models directory exists so we don't crash again!
if not os.path.exists("models"):
    os.makedirs("models")

def train():
    # 1. Setup Device
    device = cfg.device
    print(f"Using device: {device}")

    # 2. Prepare Data
    print("Loading dataset...")
    dataset = TextDataset("data/input.txt", cfg.block_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # 3. Initialize Model
    model = GPT().to(device)
    
    # Calculate model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params / 1e6:.1f} Million")

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    # 5. Training Loop
    model.train()
    print("Starting training...")
    
    data_iter = iter(dataloader)
    
    # WRAPPING THE RANGE WITH TQDM CREATES THE BAR
    # desc="Training" gives the bar a title
    progress_bar = tqdm(range(cfg.max_iters), desc="Training")
    
    for step in progress_bar:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(inputs, targets=targets)
        loss.backward()
        optimizer.step()
        
        # Update the progress bar description with the current loss
        if step % 100 == 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Save checkpoint halfway
        if step == cfg.max_iters // 2:
            # We use tqdm.write so it doesn't break the progress bar layout
            tqdm.write("Saving checkpoint...")
            torch.save(model.state_dict(), cfg.model_path)

    # 6. Save Final Model
    torch.save(model.state_dict(), cfg.model_path)
    print(f"\nTraining Complete! Model saved to {cfg.model_path}")

if __name__ == "__main__":
    train()