#!/usr/bin/env python
"""
DEBUG: Log every shape transformation
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


class DebugTeacher(nn.Module):
    def __init__(self, audio_dim=256, latent_dim=4, target_seq_len=65536):
        super().__init__()
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        self.target_seq_len = target_seq_len
        print(f"[INIT] DebugTeacher created with target_seq_len={target_seq_len}")
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.expansion_fc = nn.Linear(256, 256)
        self.latent_proj = nn.Linear(256, latent_dim)
    
    def forward(self, audio_emb):
        print(f"[FWD] Input audio_emb shape: {audio_emb.shape}")
        
        if audio_emb.ndim == 2:
            audio_emb = audio_emb.unsqueeze(1)
            print(f"[FWD] After unsqueeze: {audio_emb.shape}")
        
        B, T_audio, _ = audio_emb.shape
        print(f"[FWD] B={B}, T_audio={T_audio}, target={self.target_seq_len}")
        
        x = self.audio_encoder(audio_emb)
        print(f"[FWD] After audio_encoder: {x.shape}")
        
        # CRITICAL: Expand
        print(f"[FWD] Checking if expansion needed: {T_audio} != {self.target_seq_len}?")
        
        if T_audio != self.target_seq_len:
            print(f"[FWD] ✅ YES, expanding...")
            repeat_factor = max(1, self.target_seq_len // T_audio)
            print(f"[FWD] repeat_factor = {self.target_seq_len} // {T_audio} = {repeat_factor}")
            
            x = x.repeat(1, repeat_factor, 1)
            print(f"[FWD] After repeat: {x.shape}")
            
            if x.shape[1] != self.target_seq_len:
                print(f"[FWD] Still not exact, interpolating {x.shape[1]} -> {self.target_seq_len}")
                x_t = x.transpose(1, 2)
                print(f"[FWD] After transpose to (B,C,T): {x_t.shape}")
                
                x_t = nn.functional.interpolate(
                    x_t,
                    size=self.target_seq_len,
                    mode='linear',
                    align_corners=False
                )
                print(f"[FWD] After interpolate: {x_t.shape}")
                
                x = x_t.transpose(1, 2)
                print(f"[FWD] After transpose back: {x.shape}")
        else:
            print(f"[FWD] ❌ NO expansion needed (already correct size)")
        
        x = self.expansion_fc(x)
        print(f"[FWD] After expansion_fc: {x.shape}")
        x = torch.relu(x)
        
        latents = self.latent_proj(x)
        print(f"[FWD] After latent_proj: {latents.shape}")
        print(f"[FWD] ===== OUTPUT: {latents.shape} =====\n")
        
        return latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default="datasets/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    
    # Load VAE
    print("[MAIN] Loading VAE...")
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    try:
        vae.load_state_dict(torch.load(f"{args.checkpoint_dir}/vae_best.pt", map_location=device))
    except:
        pass
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("[MAIN] VAE loaded\n")
    
    # Load audio encoder
    print("[MAIN] Loading Audio Encoder...")
    audio_encoder = SpeechAE(audio_dim=384, output_dim=256).to(device)
    audio_encoder.eval()
    for param in audio_encoder.parameters():
        param.requires_grad = False
    print("[MAIN] Audio encoder loaded\n")
    
    # Get target seq len
    print("[MAIN] Detecting VAE output...")
    with torch.no_grad():
        dummy_video = torch.randn(1, 3, 2, 16, 16).to(device)
        _, vae_latents, _ = vae.encode(dummy_video)
        B, C, T, H, W = vae_latents.shape
        target_seq_len = T * H * W
        print(f"[MAIN] VAE shape: (B,C,T,H,W) = {vae_latents.shape}")
        print(f"[MAIN] Target seq_len: {target_seq_len}\n")
    
    # Create teacher
    print(f"[MAIN] Creating DebugTeacher with target_seq_len={target_seq_len}...\n")
    teacher = DebugTeacher(audio_dim=256, latent_dim=4, target_seq_len=target_seq_len).to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=args.lr)
    
    # Load data
    print("[MAIN] Loading data...")
    train_loader, _ = TalkingHeadDataLoader.create_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        use_dummy=True,
    )
    print("[MAIN] Data loaded\n")
    
    # Single training iteration (verbose)
    print("[MAIN] Starting ONE training iteration (verbose mode)...\n")
    
    batch = next(iter(train_loader))
    video = batch['video'].to(device)
    audio = batch['audio'].to(device)
    
    print(f"[TRAIN] Batch video shape: {video.shape}")
    print(f"[TRAIN] Batch audio shape: {audio.shape}\n")
    
    with torch.no_grad():
        _, vae_latents, _ = vae.encode(video)
        B_vae, C_vae, T_vae, H_vae, W_vae = vae_latents.shape
        latents_target = vae_latents.permute(0, 2, 3, 4, 1).reshape(B_vae, -1, C_vae)
        audio_emb = audio_encoder(audio)
        
        print(f"[TRAIN] VAE latents shape: {vae_latents.shape}")
        print(f"[TRAIN] Flattened target shape: {latents_target.shape}")
        print(f"[TRAIN] Audio embeddings shape: {audio_emb.shape}\n")
    
    optimizer.zero_grad()
    print("[TRAIN] Calling teacher.forward()...\n")
    pred_latents = teacher(audio_emb)
    
    print(f"\n[TRAIN] FINAL CHECK:")
    print(f"[TRAIN] Predicted shape: {pred_latents.shape}")
    print(f"[TRAIN] Target shape: {latents_target.shape}")
    print(f"[TRAIN] Match? {pred_latents.shape == latents_target.shape}")
    
    if pred_latents.shape == latents_target.shape:
        print(f"\n✅ SUCCESS! Shapes match. Computing loss...")
        loss = nn.MSELoss()(pred_latents, latents_target)
        print(f"Loss: {loss.item():.6f}")
    else:
        print(f"\n❌ FAILURE! Shapes don't match.")
        sys.exit(1)


if __name__ == "__main__":
    main()
