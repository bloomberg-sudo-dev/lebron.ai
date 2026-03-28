#!/usr/bin/env python
"""Diagnose actual tensor shapes at each pipeline step"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE, SpeechAE
from datasets import TalkingHeadDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
print("Loading VAE...")
vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
vae.eval()

print("Loading Audio Encoder...")
audio_encoder = SpeechAE(audio_dim=384, output_dim=256).to(device)
audio_encoder.eval()

# Load one batch
print("Loading data...")
train_loader, _ = TalkingHeadDataLoader.create_loaders(
    data_root="datasets/",
    batch_size=2,
    use_dummy=True,
)

# Get first batch
batch = next(iter(train_loader))
video = batch['video'].to(device)
audio = batch['audio'].to(device)

print(f"\n📊 INPUT SHAPES:")
print(f"  video: {video.shape}")
print(f"  audio: {audio.shape}")

# Process through VAE
with torch.no_grad():
    _, vae_latents, _ = vae.encode(video)
    print(f"\n🔴 VAE OUTPUTS:")
    print(f"  vae_latents (B,C,T,H,W): {vae_latents.shape}")
    B, C, T, H, W = vae_latents.shape
    seq_len_target = T * H * W
    print(f"  Flattened target seq_len: {T}×{H}×{W} = {seq_len_target}")
    
    # Flatten as training does
    latents_flat = vae_latents.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
    print(f"  vae_latents flattened (B,seq,C): {latents_flat.shape}")

# Process through audio encoder
with torch.no_grad():
    audio_emb = audio_encoder(audio)
    print(f"\n🔵 AUDIO ENCODER OUTPUTS:")
    print(f"  audio_emb: {audio_emb.shape}")

print(f"\n❌ MISMATCH:")
print(f"  Need teacher to output: (B={B}, seq={seq_len_target}, C={C})")
print(f"  But audio_emb is only: {audio_emb.shape}")
print(f"\n💡 Solution: Teacher must EXPAND audio_emb from {audio_emb.shape[1]} → {seq_len_target} tokens")
