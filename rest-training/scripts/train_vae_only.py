#!/usr/bin/env python
"""
VAE-only training (no teacher)
Just trains the VAE until good loss, then done.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE
from datasets import TalkingHeadDataLoader


def main():
    parser = argparse.ArgumentParser(description="Train VAE Only")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}\n")
    
    # Create VAE
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    print("✅ VAE created\n")
    
    # Data
    train_loader, _ = TalkingHeadDataLoader.create_loaders(
        data_root="datasets/",
        batch_size=args.batch_size,
        use_dummy=True,
    )
    print(f"✅ Data loader ready\n")
    
    # Train
    print(f"🚀 Training VAE for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)  # (B, 3, T, H, W)
            
            # VAE forward
            optimizer.zero_grad()
            recon, latents, _ = vae(video)
            
            # Loss: just MSE for now
            loss = nn.MSELoss()(recon, video)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f}")
    
    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(vae.state_dict(), "checkpoints/vae_best.pt")
    print(f"\n✅ Checkpoint saved: checkpoints/vae_best.pt")
    print("✨ Training complete!")


if __name__ == "__main__":
    main()
