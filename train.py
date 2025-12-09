import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.model import GPT
from src.dataset import TextDataset
from src.config import cfg
from tqdm import tqdm
import os

# Ensure the models folder exists
os.makedirs(cfg.model_save_dir, exist_ok=True)

def train():
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = TextDataset("data/input.txt", cfg.block_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    # Initialize model
    model = GPT().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params / 1e6:.1f} Million")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    
    # Gradient accumulation & mixed precision
    accumulation_steps = cfg.accumulation_steps
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    model.train()
    data_iter = iter(dataloader)
    progress_bar = tqdm(range(cfg.max_iters), desc="Training")
    accumulated_loss = 0.0
    
    # Checkpoint steps (quarter, half, final)
    checkpoint_steps = [cfg.max_iters // 4, cfg.max_iters // 2, cfg.max_iters]
    
    for step in progress_bar:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass (mixed precision if CUDA)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, loss = model(inputs, targets=targets)
                loss = loss / accumulation_steps
        else:
            logits, loss = model(inputs, targets=targets)
            loss = loss / accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        accumulated_loss += loss.item()
        
        # Optimizer step after accumulation
        if (step + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            if (step + 1) % 100 == 0:  # Show every 100 steps
                progress_bar.set_postfix(loss=f"{accumulated_loss:.4f}")
            accumulated_loss = 0.0
        
        # Save checkpoints
        if step + 1 in checkpoint_steps:
            checkpoint_path = os.path.join(cfg.model_save_dir, f"model_step{step+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"Checkpoint saved at step {step+1} → {checkpoint_path}")
    
    # Handle any remaining gradients
    if (step + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    # Final save
    final_path = os.path.join(cfg.model_save_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining Complete! Model saved to {final_path}")

if __name__ == "__main__":
    train()