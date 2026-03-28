#!/usr/bin/env python
"""
Teacher Training - WORKING VERSION
Dynamically computes sequence length per batch (not at init)
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


class DynamicTeacher(nn.Module):
    """Teacher that expands audio to match ACTUAL VAE output (computed per batch)"""
    
    def __init__(self, audio_dim=256, latent_dim=4):
        super().__init__()
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        
        # Audio processing
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Expansion (will be dynamic)
        self.expansion_fc = nn.Linear(256, 256)
        
        # Project to latent
        self.latent_proj = nn.Linear(256, latent_dim)
    
    def forward(self, audio_emb, target_seq_len):
        """
        audio_emb: (B, T_audio, audio_dim)
        target_seq_len: int (computed from actual VAE latent shape)
        output: (B, target_seq_len, latent_dim)
        """
        B, T_audio, _ = audio_emb.shape
        
        # Process audio
        x = self.audio_encoder(audio_emb)  # (B, T_audio, 256)
        
        # Expand to match VAE output
        if T_audio != target_seq_len:
            # Repeat + interpolate
            repeat_factor = max(1, target_seq_len // T_audio)
            x = x.repeat(1, repeat_factor, 1)  # (B, T*repeat, 256)
            
            # Fine-tune with interpolation
            if x.shape[1] != target_seq_len:
                x = x.transpose(1, 2)  # (B, 256, T*repeat)
                x = nn.functional.interpolate(
                    x, size=target_seq_len, mode='linear', align_corners=False
                )
                x = x.transpose(1, 2)  # (B, target_seq_len, 256)
        
        x = self.expansion_fc(x)
        x = torch.relu(x)
        
        # Project to latents
        latents = self.latent_proj(x)  # (B, target_seq_len, latent_dim)
        
        return latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default="datasets/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}\n")
    
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    
    # Load VAE (frozen)
    print("📦 Loading VAE...")
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    try:
        vae.load_state_dict(torch.load(f"{args.checkpoint_dir}/vae_best.pt", map_location=device))
    except:
        print("   (Using random init for testing)")
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("✅ VAE loaded\n")
    
    # Load audio encoder (frozen)
    print("📦 Loading Audio Encoder...")
    audio_encoder = SpeechAE(audio_dim=384, output_dim=256).to(device)
    audio_encoder.eval()
    for param in audio_encoder.parameters():
        param.requires_grad = False
    print("✅ Audio encoder loaded\n")
    
    # Create teacher (no target_seq_len needed at init)
    print("🏫 Creating Dynamic Teacher...")
    teacher = DynamicTeacher(audio_dim=256, latent_dim=4).to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=args.lr)
    print("✅ Teacher created\n")
    
    # Load data
    print("📊 Loading data...")
    train_loader, _ = TalkingHeadDataLoader.create_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        use_dummy=True,
    )
    print("✅ Data loaded\n")
    
    # Training loop
    print(f"🚀 Training for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            # Get VAE latents (frozen)
            with torch.no_grad():
                _, vae_latents, _ = vae.encode(video)
                B_vae, C_vae, T_vae, H_vae, W_vae = vae_latents.shape
                
                # DYNAMIC: Compute target_seq_len from ACTUAL VAE output
                target_seq_len = T_vae * H_vae * W_vae
                
                # Flatten VAE latents
                latents_target = vae_latents.permute(0, 2, 3, 4, 1).reshape(B_vae, -1, C_vae)
                
                # Get audio embeddings
                audio_emb = audio_encoder(audio)
            
            # Teacher forward (with DYNAMIC target_seq_len)
            optimizer.zero_grad()
            pred_latents = teacher(audio_emb, target_seq_len=target_seq_len)
            
            # Verify shapes
            assert pred_latents.shape == latents_target.shape, \
                f"Shape mismatch: {pred_latents.shape} vs {latents_target.shape}"
            
            # Loss
            loss = nn.MSELoss()(pred_latents, latents_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.6f}")
    
    # Save
    checkpoint_path = f"{args.checkpoint_dir}/teacher_working.pt"
    torch.save(teacher.state_dict(), checkpoint_path)
    print(f"\n✅ Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
