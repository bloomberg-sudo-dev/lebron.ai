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
    
    # Create teacher
    teacher = SimpleTeacher(audio_dim=256, latent_dim=4, seq_len=512).to(device)
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
                latents_flat = latents.view(latents.shape[0], -1, latents.shape[1]).transpose(1, 2)  # (B, 512, 4)
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
