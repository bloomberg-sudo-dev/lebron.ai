#!/usr/bin/env python
"""
Temporal VAE Training Script
Learns to compress video to 32x32x8 latent space
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
import sys

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE
from datasets import TalkingHeadDataLoader

def main():
    parser = argparse.ArgumentParser(description="Train Temporal VAE")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="logs/vae")
    args = parser.parse_args()
    
    print("=" * 60)
    print("REST: Temporal VAE Training")
    print("=" * 60)
    
    # Load config
    if Path(args.config).exists():
        config = OmegaConf.load(args.config)
        print(f"\n✅ Config loaded from {args.config}")
    else:
        print(f"⚠️  Config not found: {args.config}")
        config = OmegaConf.create({})
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Create dataloaders
    print(f"\n📦 Creating dataloaders...")
    # Reduce batch size to fit in GPU memory
    actual_batch_size = min(args.batch_size, 2)
    train_loader, val_loader = TalkingHeadDataLoader.create_loaders(
        data_root="datasets/",
        batch_size=actual_batch_size,
        use_dummy=True,  # Using dummy for testing
    )
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\n🧠 Creating Temporal VAE...")
    model = TemporalVAE(
        in_channels=3,
        latent_channels=4,
        hidden_dims=[128, 256, 512],
    ).to(device)
    print(f"✅ Model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    print(f"✅ Optimizer: Adam (lr=1e-4)")
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)  # (B, 3, T, H, W)
            
            # Forward pass
            optimizer.zero_grad()
            recon_video, mu, logvar = model(video)
            
            # VAE loss
            loss, recon_loss, kl_loss = model.vae_loss(recon_video, video, mu, logvar)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                recon_video, mu, logvar = model(video)
                loss, _, _ = model.vae_loss(recon_video, video, mu, logvar)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"vae_epoch_{epoch+1}.pt")
            print(f"  💾 Checkpoint saved: vae_epoch_{epoch+1}.pt")
    
    # Save final model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "vae_best.pt")
    print(f"\n✅ Training complete!")
    print(f"✅ Best model saved: checkpoints/vae_best.pt")

if __name__ == "__main__":
    main()
