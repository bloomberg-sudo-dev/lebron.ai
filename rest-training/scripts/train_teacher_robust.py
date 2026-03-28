#!/usr/bin/env python
"""
Robust Teacher Model Training - Fixed Shape Handling
Maps audio → latents with proper dimension matching
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE, SpeechAE
from datasets import TalkingHeadDataLoader


class RobustTeacher(nn.Module):
    """Teacher that properly handles shape mismatches"""
    
    def __init__(self, audio_dim=256, latent_dim=4, target_seq_len=None):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, 256)
        self.hidden = nn.Linear(256, 512)
        self.latent_proj = nn.Linear(512, latent_dim)
        self.target_seq_len = target_seq_len
        
    def forward(self, audio_emb):
        """
        audio_emb: (B, T_audio, audio_dim) or (B, audio_dim)
        returns: (B, target_seq_len, latent_dim)
        """
        # Handle 2D input
        if audio_emb.ndim == 2:
            audio_emb = audio_emb.unsqueeze(1)  # (B, 1, audio_dim)
        
        B, T_audio, _ = audio_emb.shape
        
        # Project audio features
        x = self.audio_proj(audio_emb)  # (B, T_audio, 256)
        x = torch.relu(x)
        x = self.hidden(x)  # (B, T_audio, 512)
        x = torch.relu(x)
        
        # CRITICAL: Resample to match VAE latent sequence length
        if self.target_seq_len is not None and T_audio != self.target_seq_len:
            # Transpose for interpolation: (B, 512, T_audio)
            x = x.transpose(1, 2)
            # Interpolate to target length
            x = torch.nn.functional.interpolate(
                x, 
                size=self.target_seq_len, 
                mode='linear', 
                align_corners=False
            )
            # Transpose back: (B, target_seq_len, 512)
            x = x.transpose(1, 2)
        
        # Project to latent dimension
        latents = self.latent_proj(x)  # (B, target_seq_len, latent_dim)
        
        return latents


def main():
    parser = argparse.ArgumentParser(description="Train Robust Teacher Model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default="datasets/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}\n")
    
    # Create checkpoint dir
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    
    # Load frozen VAE
    print("📦 Loading VAE...")
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    try:
        vae.load_state_dict(torch.load(f"{args.checkpoint_dir}/vae_best.pt", map_location=device))
    except FileNotFoundError:
        print(f"⚠️  VAE checkpoint not found at {args.checkpoint_dir}/vae_best.pt")
        print("   Using random initialization for testing")
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("✅ VAE loaded (frozen)\n")
    
    # Load audio encoder
    print("📦 Loading Audio Encoder...")
    audio_encoder = SpeechAE(audio_dim=384, output_dim=256).to(device)
    audio_encoder.eval()
    for param in audio_encoder.parameters():
        param.requires_grad = False
    print("✅ Audio encoder loaded (frozen)\n")
    
    # **CRITICAL: Calculate target sequence length from VAE**
    print("🔍 Detecting VAE latent sequence length...")
    with torch.no_grad():
        # Dummy video: (1, 3, 2, 16, 16) = (B, C, T, H, W)
        dummy_video = torch.randn(1, 3, 2, 16, 16).to(device)
        _, vae_latents, _ = vae.encode(dummy_video)
        
        # vae_latents shape: (B, latent_channels, T, H, W)
        B, C, T, H, W = vae_latents.shape
        target_seq_len = T * H * W
        
        print(f"   VAE latent shape (B,C,T,H,W): {vae_latents.shape}")
        print(f"   Flattened dimension: {T} × {H} × {W} = {target_seq_len}")
        print(f"✅ Target sequence length: {target_seq_len}\n")
    
    # Create teacher with correct target length
    print("🏫 Creating Teacher Model...")
    teacher = RobustTeacher(
        audio_dim=256, 
        latent_dim=4, 
        target_seq_len=target_seq_len
    ).to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=args.lr)
    print(f"✅ Teacher created with target_seq_len={target_seq_len}\n")
    
    # Load data
    print("📊 Loading training data...")
    train_loader, val_loader = TalkingHeadDataLoader.create_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        use_dummy=True,
    )
    print(f"✅ Data loaded (batch_size={args.batch_size})\n")
    
    # Training loop
    print(f"🚀 Training for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            # Get VAE latents and audio embeddings (no grad)
            with torch.no_grad():
                _, vae_latents, _ = vae.encode(video)
                B_vae, C_vae, T_vae, H_vae, W_vae = vae_latents.shape
                
                # Flatten VAE latents: (B, C, T, H, W) → (B, T*H*W, C)
                latents_target = vae_latents.permute(0, 2, 3, 4, 1).reshape(B_vae, -1, C_vae)
                
                # Get audio embeddings
                audio_emb = audio_encoder(audio)
            
            # Teacher forward pass
            optimizer.zero_grad()
            pred_latents = teacher(audio_emb)
            
            # Verify shapes match
            assert pred_latents.shape == latents_target.shape, \
                f"Shape mismatch: pred={pred_latents.shape} vs target={latents_target.shape}"
            
            # Compute loss (MSE)
            loss = nn.MSELoss()(pred_latents, latents_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_loss = train_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.6f}")
    
    # Save checkpoint
    checkpoint_path = f"{args.checkpoint_dir}/teacher_robust.pt"
    torch.save(teacher.state_dict(), checkpoint_path)
    print(f"\n✅ Checkpoint saved: {checkpoint_path}")
    
    # Save config for inference
    config = {
        'audio_dim': 256,
        'latent_dim': 4,
        'target_seq_len': target_seq_len,
    }
    torch.save(config, f"{args.checkpoint_dir}/teacher_config.pt")
    print(f"✅ Config saved: {args.checkpoint_dir}/teacher_config.pt")


if __name__ == "__main__":
    main()
