#!/usr/bin/env python
"""
Teacher Model Training - FIXED VERSION
Explicitly expands audio to match VAE latent dimensionality
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


class ExpandingTeacher(nn.Module):
    """Teacher that explicitly expands audio sequence to VAE latent size"""
    
    def __init__(self, audio_dim=256, latent_dim=4, target_seq_len=65536):
        super().__init__()
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        self.target_seq_len = target_seq_len
        
        # Stage 1: Process audio features
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        
        # Stage 2: Expand sequence
        # If input is (B, T_in, 256), expand T_in to target_seq_len
        self.expansion_fc = nn.Linear(256, 256)
        
        # Stage 3: Project to latent
        self.latent_proj = nn.Linear(256, latent_dim)
    
    def forward(self, audio_emb):
        """
        audio_emb: (B, T_audio, audio_dim) or (B, audio_dim)
        output: (B, target_seq_len, latent_dim)
        """
        # Handle 2D input (batch only, no time)
        if audio_emb.ndim == 2:
            audio_emb = audio_emb.unsqueeze(1)  # (B, 1, audio_dim)
        
        B, T_audio, _ = audio_emb.shape
        
        # Stage 1: Process audio
        x = self.audio_encoder(audio_emb)  # (B, T_audio, 256)
        
        # Stage 2: Expand sequence dimension
        # Key insight: repeat audio tokens to fill target sequence
        if T_audio != self.target_seq_len:
            # Repeat tokens to approximately match target length
            repeat_factor = max(1, self.target_seq_len // T_audio)
            x = x.repeat(1, repeat_factor, 1)  # (B, T_audio*repeat_factor, 256)
            
            # If still not exact, interpolate
            if x.shape[1] != self.target_seq_len:
                x_t = x.transpose(1, 2)  # (B, 256, T_audio*repeat_factor)
                x_t = nn.functional.interpolate(
                    x_t,
                    size=self.target_seq_len,
                    mode='linear',
                    align_corners=False
                )
                x = x_t.transpose(1, 2)  # (B, target_seq_len, 256)
        
        x = self.expansion_fc(x)  # (B, target_seq_len, 256)
        x = torch.relu(x)
        
        # Stage 3: Project to latent dimension
        latents = self.latent_proj(x)  # (B, target_seq_len, latent_dim)
        
        return latents


def main():
    parser = argparse.ArgumentParser(description="Train Teacher Model (FIXED)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default="datasets/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}\n")
    
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    
    # Load frozen VAE
    print("📦 Loading VAE...")
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    try:
        vae.load_state_dict(torch.load(f"{args.checkpoint_dir}/vae_best.pt", map_location=device))
    except FileNotFoundError:
        print(f"   ⚠️  VAE checkpoint not found, using random init for testing")
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("✅ VAE loaded (frozen)\n")
    
    # Load frozen audio encoder
    print("📦 Loading Audio Encoder...")
    audio_encoder = SpeechAE(audio_dim=384, output_dim=256).to(device)
    audio_encoder.eval()
    for param in audio_encoder.parameters():
        param.requires_grad = False
    print("✅ Audio encoder loaded (frozen)\n")
    
    # Get target sequence length from VAE
    print("🔍 Detecting VAE output dimensions...")
    with torch.no_grad():
        dummy_video = torch.randn(1, 3, 2, 16, 16).to(device)
        _, vae_latents, _ = vae.encode(dummy_video)
        B, C, T, H, W = vae_latents.shape
        target_seq_len = T * H * W
        print(f"   VAE output (B,C,T,H,W): {vae_latents.shape}")
        print(f"   Target seq_len: {T}×{H}×{W} = {target_seq_len}\n")
    
    # Create teacher
    print(f"🏫 Creating Teacher (target_seq_len={target_seq_len})...")
    teacher = ExpandingTeacher(
        audio_dim=256,
        latent_dim=4,
        target_seq_len=target_seq_len
    ).to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=args.lr)
    print("✅ Teacher created\n")
    
    # Load data
    print("📊 Loading data...")
    train_loader, _ = TalkingHeadDataLoader.create_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        use_dummy=True,
    )
    print(f"✅ Data loaded\n")
    
    # Training
    print(f"🚀 Training for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            # Get targets (frozen)
            with torch.no_grad():
                _, vae_latents, _ = vae.encode(video)
                B_vae, C_vae, T_vae, H_vae, W_vae = vae_latents.shape
                latents_target = vae_latents.permute(0, 2, 3, 4, 1).reshape(B_vae, -1, C_vae)
                
                audio_emb = audio_encoder(audio)
            
            # Teacher forward
            optimizer.zero_grad()
            pred_latents = teacher(audio_emb)
            
            # Verify shapes
            if pred_latents.shape != latents_target.shape:
                print(f"❌ Shape mismatch: {pred_latents.shape} vs {latents_target.shape}")
                print(f"   Teacher output: {pred_latents.shape}")
                print(f"   Target output: {latents_target.shape}")
                sys.exit(1)
            
            # Loss
            loss = nn.MSELoss()(pred_latents, latents_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.6f}")
    
    # Save
    checkpoint_path = f"{args.checkpoint_dir}/teacher_fixed.pt"
    torch.save(teacher.state_dict(), checkpoint_path)
    print(f"\n✅ Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
