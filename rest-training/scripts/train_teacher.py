#!/usr/bin/env python
"""
Teacher Model Training Script (Non-Streaming)
Trains A2V-DiT with VAE encoder (frozen)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE, A2VDIT
from datasets import TalkingHeadDataLoader

def main():
    parser = argparse.ArgumentParser(description="Train Teacher Model")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--vae-checkpoint", type=str, default="checkpoints/vae_best.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="logs/teacher")
    args = parser.parse_args()
    
    print("=" * 60)
    print("REST: Teacher Model Training (Non-Streaming)")
    print("=" * 60)
    
    # Load config
    if Path(args.config).exists():
        config = OmegaConf.load(args.config)
        print(f"\n✅ Config loaded from {args.config}")
    else:
        config = OmegaConf.create({})
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Load VAE
    print(f"\n🧠 Loading Temporal VAE...")
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    
    if Path(args.vae_checkpoint).exists():
        vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
        print(f"✅ VAE loaded from {args.vae_checkpoint}")
    else:
        print(f"⚠️  VAE checkpoint not found: {args.vae_checkpoint}")
        print("   Training with untrained VAE")
    
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    # Create dataloaders
    print(f"\n📦 Creating dataloaders...")
    train_loader, val_loader = TalkingHeadDataLoader.create_loaders(
        data_root="datasets/",
        batch_size=args.batch_size,
        use_dummy=True,
    )
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Create teacher model
    print(f"\n🧠 Creating A2V-DiT (Teacher)...")
    model = A2VDIT(
        latent_dim=4,
        hidden_dim=768,
        num_heads=12,
        num_blocks=28,
        audio_dim=256,
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
            audio = batch['audio'].to(device)  # (B, audio_samples)
            ref_frame = batch['ref_frame'].to(device)  # (B, 3, H, W)
            
            # Compress video with VAE
            with torch.no_grad():
                _, latents, _ = vae.encode(video)  # (B, 4, T, H', W')
            
            # Forward pass through teacher
            optimizer.zero_grad()
            output = model(latents, audio, ref_frame)
            
            # Simple MSE loss (placeholder - would use diffusion loss in real impl)
            loss = nn.MSELoss()(output, latents)
            
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
                audio = batch['audio'].to(device)
                ref_frame = batch['ref_frame'].to(device)
                
                _, latents, _ = vae.encode(video)
                output = model(latents, audio, ref_frame)
                loss = nn.MSELoss()(output, latents)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"teacher_epoch_{epoch+1}.pt")
            print(f"  💾 Checkpoint saved: teacher_epoch_{epoch+1}.pt")
    
    # Save final model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "teacher_best.pt")
    print(f"\n✅ Training complete!")
    print(f"✅ Best model saved: checkpoints/teacher_best.pt")

if __name__ == "__main__":
    main()
