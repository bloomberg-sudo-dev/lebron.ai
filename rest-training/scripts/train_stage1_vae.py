#!/usr/bin/env python3
"""
Stage 1: Temporal VAE Training
Trains the VAE on pre-computed embeddings
Input: video embeddings (B, C, T, H, W)
Output: Trained VAE checkpoint
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE
from datasets import TalkingHeadDataLoader

def train_epoch(model, loader, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        video = batch['video'].to(device)  # (B, C, T, H, W)
        
        # Forward pass
        optimizer.zero_grad()
        recon, mu, logvar = model(video)
        
        # Loss: reconstruction + KL divergence
        recon_loss = nn.functional.mse_loss(recon, video)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.001 * kl_loss  # Scale KL term
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(loader)
    avg_recon = total_recon / len(loader)
    avg_kl = total_kl / len(loader)
    
    return {
        'loss': avg_loss,
        'recon_loss': avg_recon,
        'kl_loss': avg_kl
    }

def val_epoch(model, loader, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            video = batch['video'].to(device)
            recon, mu, logvar = model(video)
            
            recon_loss = nn.functional.mse_loss(recon, video)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.001 * kl_loss
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="Train VAE Stage 1")
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/lebron.ai/rest-training/checkpoints")
    parser.add_argument("--data-dir", type=str, default="/mnt/persistent/dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    
    print("=" * 70)
    print("REST Stage 1: Temporal VAE Training")
    print("=" * 70)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    
    # Create model
    print(f"\n🏗️  Building model...")
    model = TemporalVAE(
        in_channels=4,
        latent_channels=32,
        hidden_dims=[64, 128, 256],
        kl_weight=0.00001
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Create dataloaders
    print(f"\n📦 Loading data from {data_dir}...")
    train_loader, val_loader = TalkingHeadDataLoader.create_loaders(
        data_root=str(data_dir),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\n🎯 Starting training for {args.epochs} epochs...\n")
    
    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        history['train'].append(train_metrics)
        
        # Validate
        val_loss = val_epoch(model, val_loader, device)
        history['val'].append(val_loss)
        
        scheduler.step()
        
        print(f"  Train Loss: {train_metrics['loss']:.4f} (recon: {train_metrics['recon_loss']:.4f}, kl: {train_metrics['kl_loss']:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"vae_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"  💾 Saved checkpoint: {ckpt_path.name}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = checkpoint_dir / "vae_best.pt"
            torch.save(model.state_dict(), best_ckpt)
            print(f"  🌟 New best val loss: {val_loss:.4f}")
    
    # Save final model
    final_ckpt = checkpoint_dir / "vae_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"\n✅ Training complete. Final model saved: {final_ckpt}")
    
    # Save history
    history_path = checkpoint_dir / "vae_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ History saved: {history_path}")

if __name__ == "__main__":
    main()
