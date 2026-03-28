#!/usr/bin/env python
"""
Simplified Teacher Model Training
Maps audio → latents directly (no complex attention)
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


class SimpleTeacher(nn.Module):
    """Minimal teacher: audio features → latents"""
    
    def __init__(self, audio_dim=256, latent_dim=4, seq_len=512):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, 128)
        self.latent_proj = nn.Linear(128, latent_dim)
        self.seq_len = seq_len
    
    def forward(self, audio_emb):
        # audio_emb: (B, audio_dim) or (B, T, audio_dim)
        if audio_emb.ndim == 2:
            audio_emb = audio_emb.unsqueeze(1)  # (B, 1, audio_dim)
        
        x = self.audio_proj(audio_emb)  # (B, T_audio, 128)
        x = torch.relu(x)
        
        # Interpolate to match latent sequence length
        if x.shape[1] != self.seq_len:
            x = x.transpose(1, 2)  # (B, 128, T_audio)
            x = torch.nn.functional.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
            x = x.transpose(1, 2)  # (B, seq_len, 128)
        
        latents = self.latent_proj(x)  # (B, seq_len, latent_dim)
        return latents


def main():
    parser = argparse.ArgumentParser(description="Train Simple Teacher Model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # Load VAE (frozen)
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    vae.load_state_dict(torch.load("checkpoints/vae_best.pt", map_location=device))
    vae.eval()
    print("✅ VAE loaded")
    
    # Audio encoder
    audio_encoder = SpeechAE(audio_dim=384, output_dim=256).to(device)
    audio_encoder.eval()
    print("✅ Audio encoder loaded")
    
    # Determine seq_len from actual VAE output
    print("Determining sequence length from VAE...")
    dummy_video = torch.randn(1, 3, 2, 16, 16).to(device)  # 2 frames, 16x16
    with torch.no_grad():
        _, dummy_latents, _ = vae.encode(dummy_video)
        # dummy_latents shape: (B, C, T, H, W) = (1, 4, T, H, W)
        B, C, T, H, W = dummy_latents.shape
        seq_len = T * H * W  # Total flattened sequence length
    print(f"✅ VAE output shape (B,C,T,H,W): {dummy_latents.shape}")
    print(f"✅ Flattened sequence length: {seq_len}")
    
    # Create teacher with correct seq_len
    teacher = SimpleTeacher(audio_dim=256, latent_dim=4, seq_len=seq_len).to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=args.lr)
    print("✅ Teacher model created")
    
    # Data
    train_loader, _ = TalkingHeadDataLoader.create_loaders(
        data_root="datasets/",
        batch_size=args.batch_size,
        use_dummy=True,
    )
    
    # Train
    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            # Get VAE latents
            with torch.no_grad():
                _, latents, _ = vae.encode(video)  # (B, 4, T, H, W)
                # Reshape to (B, T*H*W, 4)
                B, C, T, H, W = latents.shape
                latents_flat = latents.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # (B, T*H*W, 4)
                audio_emb = audio_encoder(audio)
            
            # Teacher forward
            optimizer.zero_grad()
            pred_latents = teacher(audio_emb)  # (B, 512, 4)
            
            # MSE loss
            loss = nn.MSELoss()(pred_latents, latents_flat)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f}")
    
    # Save
    torch.save(teacher.state_dict(), "checkpoints/teacher_simple.pt")
    print(f"\n✅ Checkpoint saved: checkpoints/teacher_simple.pt")


if __name__ == "__main__":
    main()
